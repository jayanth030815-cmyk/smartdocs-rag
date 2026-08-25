from typing import List, Dict, Any, Optional
from core.indexing.dense_index import DenseVectorStore
from core.indexing.sparse_index import SparseBM25Index

class HybridRetriever:
    """
    Coordinates Multi-Strategy Hybrid Retrieval:
    1. Runs Dense Vector Search (ChromaDB) for conceptual/semantic matches.
    2. Runs Sparse Keyword Search (BM25) for exact codes, names, and term matches.
    3. Fuses both ranked lists using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_store: DenseVectorStore,
        sparse_index: SparseBM25Index,
        rrf_k: int = 60
    ):
        self.dense_store = dense_store
        self.sparse_index = sparse_index
        self.rrf_k = rrf_k  # Standard RRF smoothing constant

    def search(
        self,
        query: str,
        top_k: int = 3,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Retrieval using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = (dense_weight / (k + rank_dense)) + (sparse_weight / (k + rank_sparse))
        """
        # Fetch candidate pools from both retrievers (grab more candidates to ensure rich overlap)
        fetch_k = top_k * 3
        dense_results = self.dense_store.search(query, top_k=fetch_k)
        sparse_results = self.sparse_index.search(query, top_k=fetch_k)

        # Dictionary to store accumulated RRF scores and metadata:
        # chunk_id -> { "chunk": dict, "rrf_score": float, "dense_rank": int, "sparse_rank": int }
        fusion_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Rankings (1-indexed)
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            if cid not in fusion_map:
                fusion_map[cid] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "dense_score": item.get("score", 0.0),
                    "sparse_score": None
                }
            fusion_map[cid]["rrf_score"] += dense_weight / (self.rrf_k + rank)

        # 2. Process Sparse BM25 Rankings (1-indexed)
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            if cid not in fusion_map:
                fusion_map[cid] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "dense_score": None,
                    "sparse_score": item.get("score", 0.0)
                }
            else:
                fusion_map[cid]["sparse_rank"] = rank
                fusion_map[cid]["sparse_score"] = item.get("score", 0.0)

            fusion_map[cid]["rrf_score"] += sparse_weight / (self.rrf_k + rank)

        # 3. Sort all candidates descending by fused RRF score
        sorted_candidates = sorted(
            fusion_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # 4. Format final output
        final_results = []
        for cand in sorted_candidates[:top_k]:
            base_chunk = cand["chunk"]
            final_results.append({
                "chunk_id": base_chunk["chunk_id"],
                "text": base_chunk["text"],
                "metadata": base_chunk.get("metadata", {}),
                "rrf_score": round(cand["rrf_score"], 5),
                "dense_rank": cand["dense_rank"],
                "sparse_rank": cand["sparse_rank"],
                "strategy": "hybrid_rrf"
            })

        return final_results
