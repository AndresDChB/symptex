import logging
import traceback

from langchain_core.messages import ToolMessage
from langgraph.graph import add_messages

from chains.custom_state import CustomState
from chains.llm import get_llm
from chains.prompts import get_prompt

logger = logging.getLogger('chain_nodes')
logger.setLevel(logging.DEBUG)
def make_load_docs_node(load_patient_docs_tool):
    async def load_docs_node(state: CustomState) -> CustomState:
        tool_calls = state.get("tool_calls", []) or []
        if not tool_calls:
            return state  # nothing to do

        tool_messages: list[ToolMessage] = []

        for tc in tool_calls:
            if tc.get("name") != "load_patient_docs":
                logger.warning("Unknown tool called: %s", tc.get("name"))
                continue

            patient_doc_md = state.get("patient_doc_md")
            file_paths = []

            for patient_doc in patient_doc_md:
                file_paths.append(patient_doc.get("file_path"))
            try:
                tool_output_text = load_patient_docs_tool.invoke()
            except Exception as e:
                logger.exception("Error running load_patient_docs tool")
                tool_output_text = f"[Error loading documents: {e}]"

            tool_messages.append(
                ToolMessage(
                    name="load_patient_docs",
                    content=tool_output_text,
                    tool_call_id=tc.get("id"),
                )
            )

        state["attach_docs"] = True
        state["tool_calls"] = []  # consumed

        # add tool messages to history for next LLM call
        state["messages"] = add_messages(state["messages"], tool_messages)

        return state
    return load_docs_node

def make_patient_model_initial(tool_list: list):
    async def patient_model_initial(state: CustomState) -> CustomState:
        model = state["model"]
        condition = state["condition"]
        talkativeness = state["talkativeness"]
        patient_details = state["patient_details"]
        patient_doc_md = state.get("patient_doc_md", [])

        prompt = get_prompt(condition, talkativeness, patient_details, patient_doc_md)

        llm_with_tools = get_llm(model).bind_tools(tool_list)
        chain = prompt | llm_with_tools

        try:
            response = await chain.ainvoke(state)
            logger.debug("Initial LLM call succeeded")

            state["messages"] = add_messages(state.get("messages", []), [response])
            state["tool_calls"] = getattr(response, "tool_calls", []) or []
            state.setdefault("attach_docs", False)
            state["hard_error"] = False
            return state

        except Exception as e:
            logger.error("Error in patient_model_initial: %s", e)
            logger.error("Traceback:\n%s", traceback.format_exc())
            fallback = {
                "role": "ai",
                "content": f"Entschuldigung, ein Fehler ist aufgetreten: {str(e)}",
            }

            state["messages"] = add_messages(state.get("messages", []), [fallback])
            state["tool_calls"] = []
            state["attach_docs"] = False
            state["hard_error"] = True

            return state
    return patient_model_initial

async def patient_model_final(state: CustomState) -> CustomState:
    model = state["model"]
    llm = get_llm(model)  # can be with or without tools now

    logger.info("Calling final patient model %s", model)
    # just pass message history
    #todo this is making the response stream twice
    response = await llm.ainvoke(state["messages"])

    state["messages"] = add_messages(state["messages"], [response])
    return state


def branching_node(state: CustomState) -> str:
    """Routing function used by add_conditional_edges."""
    if state.get("hard_error", False):
        return "abort"
    tool_calls = state.get("tool_calls", []) or []
    return "has_tool_calls" if tool_calls else "no_tool_calls"