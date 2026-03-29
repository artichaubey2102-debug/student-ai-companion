from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import numpy as np


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking
# ---------------------------------------------------------------------------

def fixed_chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 0) -> list[str]:
    """
    Splits text into chunks of a fixed token count with no overlap.
    Simple and fast. May break sentences mid-thought.

    Used in the research evaluation as the baseline strategy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Strategy 2: Overlapping chunking
# ---------------------------------------------------------------------------

def overlap_chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of a fixed size with a sliding overlap window.
    The overlap ensures that context at chunk boundaries is preserved,
    reducing the risk of splitting a concept across two chunks.

    This is the default strategy used by the production application.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Strategy 3: Semantic chunking
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1-D numpy arrays."""
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.75,
    min_chunk_chars: int = 150
) -> list[str]:
    """
    Splits text at natural topic boundaries by detecting points where
    cosine similarity between consecutive sentence embeddings drops below
    a threshold. Sentences that remain on the same topic are grouped
    together into one chunk.

    Uses the 'all-MiniLM-L6-v2' sentence-transformers model locally.
    No API cost.

    Args:
        text:                 The input text.
        similarity_threshold: Sentences below this similarity score trigger
                              a new chunk. Lower values = more chunks.
        min_chunk_chars:      Minimum character length to keep a chunk.

    Returns:
        A list of text chunks split at semantic boundaries.
    """
    from sentence_transformers import SentenceTransformer

    # Split into sentences first using simple rules
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # Load a lightweight local embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Walk through sentences and cut when similarity drops
    chunks = []
    current_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold:
            chunk_text = " ".join(current_sentences).strip()
            if len(chunk_text) >= min_chunk_chars:
                chunks.append(chunk_text)
            elif chunks:
                # Merge short dangling chunk with the previous one
                chunks[-1] = chunks[-1] + " " + chunk_text
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])

    # Add the last group
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if len(chunk_text) >= min_chunk_chars:
            chunks.append(chunk_text)
        elif chunks:
            chunks[-1] = chunks[-1] + " " + chunk_text

    return chunks


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def chunk_text(text: str, strategy: str = "overlap") -> list[str]:
    """
    Convenience function that applies one of the three chunking strategies.

    Args:
        text:     The input text to chunk.
        strategy: One of 'fixed', 'overlap', or 'semantic'.

    Returns:
        A list of text chunk strings.
    """
    if strategy == "fixed":
        return fixed_chunk(text)
    elif strategy == "overlap":
        return overlap_chunk(text)
    elif strategy == "semantic":
        return semantic_chunk(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                         f"Choose from 'fixed', 'overlap', or 'semantic'.")
