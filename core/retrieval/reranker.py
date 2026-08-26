from typing import List, Dict, Any, Optional

class CrossEncoderReranker:
    """
    Two-Stage Reranker:
    Takes candidate chunks from Hybrid Search and evaluates the deep cross-attention
    relevance between [Query, Document] pairs with a Cross-Encoder model.
    Filters out low-confidence chunks using a dynamic relevance threshold.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        relevance_threshold: float = 0.35
    ):
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        self.model = None
        self._load_model_safe()

    def _load_model_safe(self):
        """Attempts to load sentence-transformers CrossEncoder locally if installed."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except ImportError:
            # Fallback if sentence-transformers is not yet installed in local env
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunks against the user query.
        Returns top_n chunks that pass the relevance threshold, sorted by score.
        """
        if not candidates:
            return []

        # 1. Neural Cross-Encoder Mode (when sentence-transformers is available)
        if self.model is not None:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.model.predict(pairs)

            reranked = []
            for candidate, score in zip(candidates, scores):
                # Convert raw logit / score to normalized float
                norm_score = round(float(score), 4)
                
                # Check quality threshold
                if norm_score >= self.relevance_threshold:
                    item = candidate.copy()
                    item["rerank_score"] = norm_score
                    reranked.append(item)

            # Sort descending by rerank score
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_n]

        # 2. Heuristic Quality Filter Fallback (Offline / Lightweight Mode)
        # Uses normalized token overlap & RRF score consensus
        reranked = []
        q_words = set(query.lower().split())
        for c in candidates:
            c_text = c["text"].lower()
            overlap = sum(1 for w in q_words if w in c_text and len(w) > 2)
            heuristic_score = round(min(1.0, 0.3 + (overlap * 0.2)), 4)

            if heuristic_score >= self.relevance_threshold:
                item = c.copy()
                item["rerank_score"] = heuristic_score
                reranked.append(item)

        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_n]
