import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from pathlib import Path
from ingest import ingest_pdf
from retrieval import query_by_image_url, query_by_image_file

app = FastAPI(
    title="Citrus Image RAG API",
    description="Upload a PDF to ingest, then query by uploading an image",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_OUTPUT_DIR = os.getenv("IMAGES_OUTPUT_DIR", "extracted_images")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

Path(IMAGES_OUTPUT_DIR).mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_OUTPUT_DIR), name="images")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "device": "cpu"}


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest-pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    """Upload a PDF file to extract images and ingest into Pinecone."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        total = ingest_pdf(pdf_path=tmp_path)

        return JSONResponse(content={
            "message": "Ingestion complete",
            "pdf_filename": file.filename,
            "total_vectors_upserted": total
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Inference ─────────────────────────────────────────────────────────────────

@app.post("/query-image-upload")
async def query_image_by_upload(file: UploadFile = File(...), top_k: int = 5):
    """Upload an image to find similar images from the knowledge base."""
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        results = query_by_image_file(image_path=tmp_path, top_k=top_k)

        for r in results:
            r["image_url"] = f"{BASE_URL}/images/{r['filename']}"

        return JSONResponse(content={
            "query_filename": file.filename,
            "top_k": top_k,
            "results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query-image-url")
async def query_image_by_url(image_url: str, top_k: int = 5):
    """Query by providing an image URL."""
    try:
        results = query_by_image_url(image_url=image_url, top_k=top_k)
        for r in results:
            r["image_url"] = f"{BASE_URL}/images/{r['filename']}"
        return JSONResponse(content={
            "query_image_url": image_url,
            "top_k": top_k,
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))