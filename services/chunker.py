import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)

def chunk_text(text: str, size=500, overlap=50):
    sentences = sent_tokenize(text)
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) > size:
            chunks.append(current.strip())
            current = current[-overlap:] + s
        else:
            current += " " + s

    if current.strip():
        chunks.append(current.strip())

    return chunks
