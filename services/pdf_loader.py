import pdfplumber

def load_pdf_content(path: str) -> str:
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

    return "\n\n".join(content)
