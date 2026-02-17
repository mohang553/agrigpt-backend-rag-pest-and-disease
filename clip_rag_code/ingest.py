import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from extract_pdf import extract_images_from_pdf

load_dotenv()

# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP_DIM = 768
CLIP_DIM = 512
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "citrus-image-rag")
IMAGES_OUTPUT_DIR = os.getenv("IMAGES_OUTPUT_DIR", "extracted_images")

print("Loading CLIP model...")
device = "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model = model.to(device)
model.eval()
print("CLIP model loaded.")


def get_image_embedding(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        image_features = output.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()


def ingest_pdf(pdf_path: str):
    print(f"\nExtracting images from: {pdf_path}")
    image_metadata = extract_images_from_pdf(pdf_path, IMAGES_OUTPUT_DIR)

    if not image_metadata:
        print("No images found in PDF.")
        return 0

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=CLIP_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)

    batch = []
    total_upserted = 0

    for i, meta in enumerate(image_metadata):
        try:
            print(f"Embedding [{i + 1}/{len(image_metadata)}]: {meta['filename']}")
            embedding = get_image_embedding(meta["filepath"])

            batch.append({
                "id": meta["image_id"],
                "values": embedding.tolist(),
                "metadata": {
                    "filename": meta["filename"],
                    "page_number": meta["page_number"],
                    "surrounding_text": meta["surrounding_text"],
                    "filepath": meta["filepath"],
                }
            })

            if len(batch) == 50:
                index.upsert(vectors=batch)
                total_upserted += len(batch)
                print(f"Upserted batch. Total so far: {total_upserted}")
                batch = []

        except Exception as e:
            print(f"Error processing {meta['filename']}: {e}")

    if batch:
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    print(f"\nIngestion complete. Total vectors upserted: {total_upserted}")
    return total_upserted