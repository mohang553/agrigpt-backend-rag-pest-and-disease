import fitz
import os
from pathlib import Path


def extract_images_from_pdf(pdf_path: str, output_dir: str = "extracted_images") -> list[dict]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_metadata = []
    seen_xrefs = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        page_text = page.get_text("text").strip()

        for img_idx, img in enumerate(image_list):
            xref = img[0]

            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                if len(image_bytes) < 5000:
                    continue

                filename = f"page{page_num + 1}_img{img_idx + 1}.{ext}"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                image_metadata.append({
                    "image_id": f"page{page_num + 1}_img{img_idx + 1}",
                    "filepath": filepath,
                    "filename": filename,
                    "page_number": page_num + 1,
                    "surrounding_text": page_text[:600],
                })

                print(f"Extracted: {filename} (page {page_num + 1})")

            except Exception as e:
                print(f"Error extracting image xref {xref} on page {page_num + 1}: {e}")

    doc.close()
    print(f"\nTotal images extracted: {len(image_metadata)}")
    return image_metadata