import logging
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from chains.chain_nodes import patient_model_initial, load_docs_node, patient_model_final, branching_node

# Load env variables for LangSmith to work
load_dotenv()

# Set up logging
logger = logging.getLogger('chat_chain')
logger.setLevel(logging.DEBUG)

class CustomState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model: str
    condition: str
    talkativeness: str
    patient_details: str
    patient_doc_md: list[dict]
    tool_calls: list
    attach_docs: bool
    hard_error: bool
        
# Set up env variables
CHATAI_API_URL = os.environ.get("CHATAI_API_URL")
CHATAI_API_KEY = os.environ.get("CHATAI_API_KEY")
if not CHATAI_API_URL or not CHATAI_API_KEY:
    logger.error("CHATAI environment variable not set, setting to default")
    raise ValueError("ERROR: Environment variables not set")

def get_llm(model: str) -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        api_key=CHATAI_API_URL,
        base_url=CHATAI_API_KEY,
        model=model,
        temperature=0.7,
        top_p=0.8,
        #max_tokens=1024,
        max_retries=2,
    )

#todo consider adding ls.traceable to LLM-calling nodes

workflow = StateGraph(state_schema=CustomState)

workflow.add_node("patient_model_initial", patient_model_initial)
workflow.add_node("load_docs", load_docs_node)
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
symptex_model = workflow.compile()
