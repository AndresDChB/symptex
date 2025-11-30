from langchain_core.tools import tool

from app.utils.pdf_utils import parse_pdf


def make_load_patient_files_tool(file_paths: list[str]):
    @tool("load_patient_docs")
    def load_patient_docs_tool() -> str:
        """
        Loads and returns the full text of all available medical documents
        (e.g., doctor's letters, reports, imaging findings, lab results).

        The Orchestrator Assistant should call this tool when the doctor asks about
        “Befund”, “Befunde”, reports, findings, test results, imaging, or any related concept.

        After calling this tool:
        - You MUST generate a short plaintext summary of the returned text.
        - Be strictly EXTRACTIVE: only include information explicitly written in
          the document. No guessing, no added details, no medical inference.
        - If something is not stated, treat it as unknown.
        """
        parts = []
        for fp in file_paths:
            text = parse_pdf(fp)
            parts.append(f"--- BEGIN DOCUMENT: {fp} ---\n{text}\n--- END DOCUMENT: {fp} ---")
        return "\n\n".join(parts)
    return load_patient_docs_tool