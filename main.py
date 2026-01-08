import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.env_utils import get_env
from utils.logger import logger

from services.pdf_loader import load_pdf_content
from services.chunker import chunk_text
from services.indexer import build_indexes
from services.retriever import hybrid_retrieve, rerank
from services.llm import generate_answer

# Environment configuration

DATA_DIR = get_env("DATA_DIR")
TOP_K = int(get_env("TOP_K_RETRIEVAL") or 10)

os.makedirs(DATA_DIR, exist_ok=True)


# FastAPI App initialization
app = FastAPI(title="RAG Service API")


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Model
class RAGResponse(BaseModel):
    status: str
    filename: Optional[str] = None
    chunks_created: Optional[int] = None
    question: Optional[str] = None
    answer: Optional[str] = None



# RAG Endpoint
@app.post("/rag", response_model=RAGResponse)
async def rag(
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    logger.info("Received RAG request")

    if not file and not question:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'file' or 'question' must be provided"
        )

    response = {"status": "success"}

    try:
        # PDF Upload & Index
        if file:
            logger.info("Processing uploaded file: %s", file.filename)

            file_path = os.path.join(DATA_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            text = load_pdf_content(file_path)
            chunks = chunk_text(text)

            if not chunks:
                raise ValueError("No text could be extracted from the PDF")

            build_indexes(chunks)

            response.update(
                {
                    "filename": file.filename,
                    "chunks_created": len(chunks),
                }
            )

            logger.info("Indexing completed with %d chunks", len(chunks))

        # Question Answering
        if question:
            logger.info("Processing question: %s", question)

            # Stage 1: Hybrid retrieval (BM25 + TF-IDF fusion)
            candidates = hybrid_retrieve(question)

            if not candidates:
                logger.warning("No retrieval candidates found")
                answer = "I don't know. No relevant context found."
            else:
                # Stage 2: Deterministic reranking
                ranked_contexts = rerank(question, candidates)[:TOP_K]

                # Final answer generation (LLM only)
                answer = generate_answer(question, ranked_contexts)

            response.update(
                {
                    "question": question,
                    "answer": answer,
                }
            )

        return response

    except FileNotFoundError as e:
        logger.error("File not found: %s", str(e))
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        logger.exception("Unexpected server error")
        raise HTTPException(status_code=500, detail="Internal server error")
