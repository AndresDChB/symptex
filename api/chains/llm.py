import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger('llm')
logger.setLevel(logging.DEBUG)

# Load env variables for LangSmith to work
load_dotenv()

# Set up env variables
CHATAI_API_URL = os.environ.get("CHATAI_API_URL")
CHATAI_API_KEY = os.environ.get("CHATAI_API_KEY")

if not CHATAI_API_URL or not CHATAI_API_KEY:
    logger.error("CHATAI environment variable not set, setting to default")
    raise ValueError("ERROR: Environment variables not set")

def get_llm(model: str) -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        api_key=CHATAI_API_KEY,
        base_url=CHATAI_API_URL,
        model=model,
        temperature=0.7,
        top_p=0.8,
        #max_tokens=1024,
        max_retries=2,
    )

