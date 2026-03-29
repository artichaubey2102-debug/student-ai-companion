def table_to_markdown(rows: list[list]) -> str:
    """
    Converts a pdfplumber table (list of row lists) into a
    markdown-formatted table string suitable for embedding.

    Handles None cells and cleans whitespace.
    Returns an empty string if the table has no usable rows.
    """
    if not rows:
        return ""

    # Clean each cell: replace None with empty string, strip whitespace
    cleaned = []
    for row in rows:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        cleaned.append(cleaned_row)

    # Determine column count from the widest row
    col_count = max(len(row) for row in cleaned)

    # Pad rows that have fewer columns than the widest row
    padded = [row + [""] * (col_count - len(row)) for row in cleaned]

    # First row becomes the header
    header = padded[0]
    separator = ["---"] * col_count
    body = padded[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def tables_to_text_blocks(tables: list[list[list]]) -> list[str]:
    """
    Converts a list of tables (as returned by pdf_extractor) into
    a list of markdown strings, one per table.
    Skips tables that produce no output.
    """
    blocks = []
    for table in tables:
        md = table_to_markdown(table)
        if md.strip():
            blocks.append(md)
    return blocks
