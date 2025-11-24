from langchain_core.tools import tool

from app.utils.pdf_utils import parse_pdf


def make_load_patient_files_tool(file_paths: list[str]):
    #todo update prompt, dont include
    @tool("load_patient_docs")
    def load_patient_docs_tool() -> str:
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
    return load_patient_docs_tool