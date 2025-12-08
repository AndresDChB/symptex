import base64
import json
import logging
import os
from pathlib import Path

import pymupdf

from app.db.models import AnamDoc

logger = logging.getLogger(__name__)
container_dir = os.environ["ANAMNESIS_DIR"]
def anamdoc_to_dict(doc: AnamDoc) -> dict:
    return {
        "id": doc.id,
        "file_path": doc.file_path,
        "type": doc.type,
        "patient_file_id": doc.patient_file_id,
        "description": doc.description,
    }

def anamdocs_to_json(anamdocs):
    dicts = [anamdoc_to_dict(doc) for doc in anamdocs]
    return json.dumps(dicts, indent=2, ensure_ascii=False)

def parse_pdf(pdf_path : str) -> str:
    doc = pymupdf.open(container_dir + pdf_path)
    all_pages_text = []

    for page_index in range(doc.page_count):
        page = doc[page_index]

        page_text = page.get_text("text")

        images = page.get_images(full=True)

        for img_index, _ in enumerate(images, start=1):
            page_text += f"\n[IMAGE {img_index} ON PAGE {page_index + 1}]\n"

        page_output = f"--- PAGE {page_index + 1} ---\n{page_text}"
        all_pages_text.append(page_output)

    doc.close()
    return "\n\n".join(all_pages_text)

def load_pdfs_as_base64(file_paths: list[str]) -> list[dict]:
    """
    Load PDFs from disk and return a list of dicts with filename + base64 content.
    """
    docs: list[dict] = []
    for path in file_paths:
        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            b64 = base64.b64encode(pdf_bytes).decode("ascii")
            docs.append({
                "filename": Path(path).name,
                "content_b64": b64,
            })
        except Exception as e:
            # optionally log and skip problematic files
            logger.error("Error loading PDF %s: %s", path, e)
    return docs
