import logging

from langgraph.graph import START, StateGraph, END

from chains.chain_nodes import patient_model_final, branching_node, make_patient_model_initial, make_load_docs_node
from chains.chain_tools import make_load_patient_files_tool
from chains.custom_state import CustomState

# Set up logging
logger = logging.getLogger('chat_chain')
logger.setLevel(logging.DEBUG)
#todo consider adding ls.traceable to LLM-calling nodes

def build_symptex_model(initial_state: CustomState):

    workflow = StateGraph(state_schema=CustomState)

    load_patient_docs_tool = make_load_patient_files_tool(extract_file_path(initial_state["patient_doc_md"]))
    workflow.add_node("patient_model_initial", make_patient_model_initial([load_patient_docs_tool]))
    workflow.add_node("load_docs", make_load_docs_node(load_patient_docs_tool))
    workflow.add_node("patient_model_final", patient_model_final)

    workflow.add_edge(START, "patient_model_initial")
    workflow.add_conditional_edges(
        "patient_model_initial",
        branching_node,
        {
            "abort": END,
            "has_tool_calls": "load_docs",
            "no_tool_calls": "patient_model_final",
        },
    )
    workflow.add_edge("load_docs", "patient_model_final")
    workflow.add_edge("patient_model_final", END)
    return workflow.compile()

def extract_file_path(patient_doc_md: list[dict]) -> list[str]:
    file_paths = []
    for patient_doc in patient_doc_md:
        file_paths.append(patient_doc.get("file_path"))
    return file_paths