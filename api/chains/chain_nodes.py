import logging
import re
import traceback

from langchain_core.messages import ToolMessage
from langgraph.graph import add_messages

import chains.prompts.orchestrator_prompts as orchestrator_prompts
import chains.prompts.patient_prompts as patient_prompts
from chains.custom_state import CustomState
from chains.llm import get_llm

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

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
                tool_output_text = load_patient_docs_tool.invoke({})
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

def make_orchestrator_node(tool_list: list):
    async def orchestrator_node(state: CustomState) -> CustomState:
        model = state["model"]
        prompt = orchestrator_prompts.get_prompt()
        logger.info("Starting orchestrator node execution")
        llm_with_tools = get_llm(model).bind_tools(tool_list)
        logger.info("Tools bound to orchestrator")
        chain = prompt | llm_with_tools

        try:
            response = await chain.ainvoke(state)
            logger.debug("Initial LLM call succeeded")
            tool_calls = getattr(response, "tool_calls", []) or []
            state["tool_calls"] = tool_calls

            if tool_calls:
                logger.debug("Orchestrator requested tool calls: %s", tool_calls)
                state["hard_error"] = False
                return state

            content = getattr(response, "content", "") or ""
            stripped = strip_think_tags(content).strip()

            if stripped == "NO_TOOL":
                logger.debug("Orchestrator returned NO_TOOL (no-op).")
                state["hard_error"] = False
                return state

            logger.debug("Orchestrator produced summary: %s", stripped)
            state["docs_summary"] = stripped
            state["attach_docs"] = True
            state["hard_error"] = False

            return state

        except Exception as e:
            logger.error("Error in orchestrator_node: %s", e)
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
    return orchestrator_node

async def patient_model_final(state: CustomState) -> CustomState:
    model = state["model"]
    condition = state["condition"]
    talkativeness = state["talkativeness"]
    patient_details = state["patient_details"]
    patient_doc_md = state.get("patient_doc_md", [])
    docs_summary = state.get("docs_summary", [])
    llm = get_llm(model)
    logger.info("Calling final patient model %s", model)
    # todo instead of this you could just leave the variables in the prompt empty and let chain.invoke() handle that
    prompt = patient_prompts.get_prompt(condition, talkativeness, patient_details, patient_doc_md, docs_summary)

    chain = prompt | llm
    response = await chain.ainvoke(state)

    state["messages"] = add_messages(state["messages"], [response])
    return state


def branching_node(state: CustomState) -> str:
    """Routing function used by add_conditional_edges."""
    if state.get("hard_error", False):
        return "abort"
    tool_calls = state.get("tool_calls", []) or []
    return "has_tool_calls" if tool_calls else "no_tool_calls"