import json

import pymupdf

from app.db.models import AnamDoc

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
    doc = pymupdf.open(pdf_path)
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
