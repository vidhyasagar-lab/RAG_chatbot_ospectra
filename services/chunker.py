import nltk
from nltk.tokenize import sent_tokenize
from utils.logger import logger

nltk.download("punkt", quiet=True)


def chunk_text(text: str, size=500, overlap=50):
    logger.info("Starting text chunking")

    try:
        sentences = sent_tokenize(text)
        chunks, current = [], ""

        for sentence in sentences:
            if len(current) + len(sentence) > size:
                chunks.append(current.strip())
                current = current[-overlap:] + sentence
            else:
                current += " " + sentence

        if current.strip():
            chunks.append(current.strip())

        logger.info("Text chunking completed. Total chunks: %d", len(chunks))
        return chunks

    except Exception as e:
        logger.error("Error while chunking text: %s", str(e))
        raise
