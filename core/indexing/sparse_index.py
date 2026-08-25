import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class SparseBM25Index:
    """
    In-memory BM25 index for exact keyword, acronym, code, and entity matching.
    """

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Cleans and tokenizes text into lowercase alphanumeric tokens.
        Preserves hyphens and underscores for codes (e.g., 'ERR-404', 'API_V2').
        """
        # Split on whitespace and punctuation, keeping alphanumeric and hyphens/underscores
        tokens = re.findall(r'[\w\-]+', text.lower())
        return [t for t in tokens if len(t) > 1]

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Indexes a list of chunk dictionaries:
        [
            {"chunk_id": "...", "text": "...", "metadata": {...}}
        ]
        """
        if not chunks:
            return

        self.chunks.extend(chunks)
        new_tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        self.corpus_tokens.extend(new_tokenized_docs)

        # Rebuild BM25 index over the entire corpus
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index and returns top_k matching chunks with scores.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Get raw BM25 scores across all documents
        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with chunk objects
        scored_chunks = []
        for idx, score in enumerate(scores):
            if score > 0.0:  # Only consider documents with at least one matching term
                c = self.chunks[idx]
                scored_chunks.append({
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                    "score": round(float(score), 4)
                })

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def clear(self):
        """Resets the index."""
        self.chunks = []
        self.corpus_tokens = []
        self.bm25 = None
