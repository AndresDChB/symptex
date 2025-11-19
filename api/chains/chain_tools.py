from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.pdf_utils import parse_pdf

class LoadPatientDocsInput(BaseModel):
    """Args for the load_patient_docs tool."""
    file_paths: list[str] = Field(
        ...,
        description=(
            "List of file paths (from patient_doc_md) of patient documents to load. "
            "Each path must exactly match a 'file_path' in patient_doc_md."
        ),
    )


@tool("load_patient_docs", args_schema=LoadPatientDocsInput)
def load_patient_docs_tool(file_paths: list[str]) -> str:
    """
    Tool that opens the given file paths (PDFs) and returns their text content
    so the LLM becomes aware of what's inside the documents.

    The backend will use the *same* file paths (stored in state) to actually
    send the PDFs to the user if desired.
    """
    parts = []
    for fp in file_paths:
        text = parse_pdf(fp)
        parts.append(f"--- BEGIN DOCUMENT: {fp} ---\n{text}\n--- END DOCUMENT: {fp} ---")
    return "\n\n".join(parts)