import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, logger
from pydantic import BaseModel

from utils.env_utils import get_env
from services.pdf_loader import load_pdf_content
from services.chunker import chunk_text
from services.indexer import build_indexes
from services.retriever import hybrid_retrieve
from services.llm import generate_answer

# Environment configuration
DATA_DIR = get_env("DATA_DIR")
os.makedirs(DATA_DIR, exist_ok=True)

# FastAPI App initialization
app = FastAPI(title="RAG Service API")


# Response body model
class RAGResponse(BaseModel):
    status: str
    filename: Optional[str] = None
    chunks_created: Optional[int] = None
    question: Optional[str] = None
    answer: Optional[str] = None


@app.post("/rag", response_model=RAGResponse)
async def rag(
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    logger.info("Received rag request")

    # Validation
    if file is None and question is None:
        logger.info("No file or question provided")
        raise HTTPException(
            status_code=400,
            detail="At least one of 'file' or 'question' must be provided"
        )

    response = {"status": "success"}

    try:
        # File handling
        if file:
            logger.info(f"Uploading file: {file.filename}")

            file_path = os.path.join(DATA_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            logger.info("Loading PDF content")
            text = load_pdf_content(file_path)

            logger.info("Chunking document")
            chunks = chunk_text(text)

            logger.info(f"Building indexes for {len(chunks)} chunks")
            build_indexes(chunks)

            response.update(
                {
                    "filename": file.filename,
                    "chunks_created": len(chunks),
                }
            )
        # Question answering
        if question:
            logger.info(f"Processing question: {question}")

            contexts = hybrid_retrieve(question)
            logger.info(f"Retrieved {len(contexts)} contexts")

            answer = generate_answer(question, contexts)

            response.update(
                {
                    "question": question,
                    "answer": answer,
                }
            )

        logger.info("Request completed successfully")
        return response

    except FileNotFoundError as e:
        logger.error("File not found error", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except ValueError as e:
        logger.error("Value error", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        logger.exception("Unexpected server error")
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )