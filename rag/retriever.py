"""
retriever.py
------------
Retrieves relevant chunks from ChromaDB using two similarity methods:

    cosine  -- standard dense retrieval, returns the top-k most similar chunks
    mmr     -- Maximal Marginal Relevance, balances relevance with diversity
               to avoid returning multiple chunks that say the same thing

Both methods accept an optional metadata filter dict produced by
query_detector.build_chroma_filter() to scope retrieval to a specific
course or unit before running the vector search.

The production application uses 'overlap' chunking collection by default.
The evaluation scripts call both methods against all three collections.
"""

import os
import numpy as np
import chromadb
from ingestion.embedder import embed_query


_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
_DEFAULT_TOP_K = 5


def _get_collection(strategy: str = "overlap") -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection_name = f"chunks_{strategy}"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_cosine(
    query: str,
    chroma_filter: dict = None,
    top_k: int = _DEFAULT_TOP_K,
    strategy: str = "overlap"
) -> list[dict]:
    """
    Retrieves the top-k chunks whose embeddings are most similar to the
    query embedding using cosine similarity.

    Args:
        query:         The student's question as a plain string.
        chroma_filter: Optional metadata filter from build_chroma_filter().
        top_k:         Number of chunks to retrieve.
        strategy:      Which ChromaDB collection to search.

    Returns:
        A list of dicts, each containing 'text', 'metadata', and 'score'.
        Ordered from most to least similar.
    """
    query_embedding = embed_query(query)
    if not query_embedding:
        return []

    collection = _get_collection(strategy)

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"]
    }

    if chroma_filter:
        query_params["where"] = chroma_filter

    try:
        results = collection.query(**query_params)
    except Exception as e:
        print(f"Warning: ChromaDB query failed. Reason: {e}")
        return []

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance = 1 - cosine similarity
        score = 1.0 - dist
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": round(score, 4)
        })

    return chunks


def retrieve_mmr(
    query: str,
    chroma_filter: dict = None,
    top_k: int = _DEFAULT_TOP_K,
    fetch_k: int = 20,
    lambda_mult: float = 0.6,
    strategy: str = "overlap"
) -> list[dict]:
    """
    Retrieves chunks using Maximal Marginal Relevance (MMR).

    MMR fetches a larger candidate pool first (fetch_k chunks), then
    iteratively selects chunks that are both relevant to the query and
    different from the chunks already selected. This reduces redundancy
    when multiple chunks from the same section of a textbook score highly.

    Args:
        query:         The student's question.
        chroma_filter: Optional metadata filter.
        top_k:         Number of chunks to return after MMR reranking.
        fetch_k:       Size of the initial candidate pool before MMR.
                       Should be larger than top_k, typically 3-4x.
        lambda_mult:   Balance between relevance (1.0) and diversity (0.0).
                       0.6 gives a reasonable balance for academic Q&A.
        strategy:      Which ChromaDB collection to search.

    Returns:
        A list of dicts with 'text', 'metadata', and 'score'.
    """
    query_embedding = embed_query(query)
    if not query_embedding:
        return []

    collection = _get_collection(strategy)
    fetch_k = min(fetch_k, collection.count())

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": fetch_k,
        "include": ["documents", "metadatas", "distances", "embeddings"]
    }

    if chroma_filter:
        query_params["where"] = chroma_filter

    try:
        results = collection.query(**query_params)
    except Exception as e:
        print(f"Warning: ChromaDB MMR query failed. Reason: {e}")
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    embeddings = results.get("embeddings", [[]])[0]

    if not documents:
        return []

    # MMR selection loop
    candidate_scores = [1.0 - d for d in distances]
    selected_indices = []
    remaining_indices = list(range(len(documents)))

    for _ in range(min(top_k, len(documents))):
        best_idx = None
        best_mmr_score = float("-inf")

        for idx in remaining_indices:
            relevance = candidate_scores[idx]

            if not selected_indices:
                redundancy = 0.0
            else:
                similarities_to_selected = [
                    _cosine_similarity(embeddings[idx], embeddings[sel])
                    for sel in selected_indices
                ]
                redundancy = max(similarities_to_selected)

            mmr_score = (
                lambda_mult * relevance - (1 - lambda_mult) * redundancy
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [
        {
            "text": documents[i],
            "metadata": metadatas[i],
            "score": round(candidate_scores[i], 4)
        }
        for i in selected_indices
    ]


def retrieve(
    query: str,
    chroma_filter: dict = None,
    method: str = "cosine",
    top_k: int = _DEFAULT_TOP_K,
    strategy: str = "overlap"
) -> list[dict]:
    """
    Unified retrieval entry point used by rag_engine.py.

    Args:
        query:         The student's question.
        chroma_filter: Optional metadata filter.
        method:        'cosine' or 'mmr'.
        top_k:         Number of chunks to return.
        strategy:      ChromaDB collection to search.

    Returns:
        List of chunk dicts with 'text', 'metadata', and 'score'.
    """
    if method == "cosine":
        return retrieve_cosine(query, chroma_filter, top_k, strategy)
    elif method == "mmr":
        return retrieve_mmr(query, chroma_filter, top_k, strategy=strategy)
    else:
        raise ValueError(
            f"Unknown retrieval method: '{method}'. "
            "Choose 'cosine' or 'mmr'."
        )
