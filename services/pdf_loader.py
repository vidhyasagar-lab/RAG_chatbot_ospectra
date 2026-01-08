import pdfplumber
from utils.logger import logger


def load_pdf_content(path: str) -> str:
    logger.info("Loading PDF content: %s", path)

    try:
        content = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []

                table_text = []
                for table in tables:
                    for row in table:
                        table_text.append(" | ".join(cell or "" for cell in row))

                content.append(
                    f"""--- Page {i} ---
TEXT:
{text}

TABLES:
{chr(10).join(table_text)}
"""
                )

        logger.info("PDF loaded successfully. Total pages: %d", len(content))
        return "\n\n".join(content)

    except Exception as e:
        logger.error("Error while loading PDF '%s': %s", path, str(e))
        raise
