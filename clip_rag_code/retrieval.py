import os
import torch
import httpx
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "citrus-image-rag-clip-vit-base-patch32-cpu")

print("Loading CLIP model for inference...")
device = "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model = model.to(device)
model.eval()
print("CLIP model ready.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Lazy index connection — only connect when needed
_index = None

def get_index():
    global _index
    if _index is None:
        _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def get_image_embedding_from_pil(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        image_features = output.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten().tolist()


def _format_results(pinecone_results) -> list[dict]:
    retrieved = []
    for match in pinecone_results["matches"]:
        retrieved.append({
            "image_id": match["id"],
            "similarity_score": round(match["score"], 4),
            "filename": match["metadata"].get("filename"),
            "page_number": match["metadata"].get("page_number"),
            "surrounding_text": match["metadata"].get("surrounding_text"),
            "filepath": match["metadata"].get("filepath"),
        })
    return retrieved


def query_by_image_url(image_url: str, top_k: int = 5) -> list[dict]:
    response = httpx.get(image_url, timeout=15)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    query_vector = get_image_embedding_from_pil(image)
    results = get_index().query(vector=query_vector, top_k=top_k, include_metadata=True)
    return _format_results(results)


def query_by_image_file(image_path: str, top_k: int = 5) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    query_vector = get_image_embedding_from_pil(image)
    results = get_index().query(vector=query_vector, top_k=top_k, include_metadata=True)
    return _format_results(results)