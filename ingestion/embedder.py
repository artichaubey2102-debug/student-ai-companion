"""
embedder.py
-----------
Generates text embeddings using a local sentence-transformers model.

Uses all-MiniLM-L6-v2 which runs entirely on your Mac with no API calls,
no quota limits, and no internet connection required after the one-time
model download (~90MB).

The model is cached locally by the sentence-transformers library after
first use. Subsequent runs load it from disk in a few seconds.

Embedding dimension: 384
"""

import os
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model_instance = None


def _get_model() -> SentenceTransformer:
    """
    Loads the embedding model once and reuses it for the session.
    Avoids reloading the model for every batch.
    """
    global _model_instance
    if _model_instance is None:
        print(f"  Loading embedding model '{_MODEL_NAME}'...")
        _model_instance = SentenceTransformer(_MODEL_NAME)
        print(f"  Embedding model loaded.")
    return _model_instance


def embed_text(text: str) -> list[float]:
    """
    Embeds a single string and returns a vector as a list of floats.
    Returns an empty list if embedding fails.
    """
    try:
        model = _get_model()
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        print(f"  Warning: embedding failed for one chunk. Reason: {e}")
        return []


def embed_query(query_text: str) -> list[float]:
    """
    Embeds a query string for retrieval.
    Uses the same model as document embedding for consistency.
    """
    return embed_text(query_text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts in one batch call.
    sentence-transformers handles batching internally and is
    significantly faster than embedding one text at a time.
    """
    total = len(texts)
    if total == 0:
        return []

    try:
        model = _get_model()
        print(f"  Embedding {total} chunks locally...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True
        )
        print(f"  Embedding complete.")
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"  Warning: batch embedding failed. Reason: {e}")
        # Fall back to one at a time if batch fails
        results = []
        for i, text in enumerate(texts):
            results.append(embed_text(text))
            if (i + 1) % 100 == 0:
                print(f"  Embedded {i + 1}/{total} chunks...")
        return results
