import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

class DenseVectorStore:
    """
    Manages vector storage and semantic search using ChromaDB.
    Uses Sentence Transformers (all-MiniLM-L6-v2) by default to generate
    384-dimensional embeddings locally on CPU/GPU without needing external API keys.
    """

    def __init__(
        self,
        collection_name: str = "smartdocs_collection",
        persist_directory: Optional[str] = None
    ):
        # By default, persist to a local folder named 'chroma_db'
        if persist_directory is None:
            persist_directory = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")

        os.makedirs(persist_directory, exist_ok=True)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use default lightweight, fast embedding function (all-MiniLM-L6-v2)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Use Cosine Similarity for distance
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Adds a list of chunk dictionaries to ChromaDB:
        chunks format:
        [
            {
                "chunk_id": "chunk_1",
                "text": "...",
                "metadata": {"source": "manual.pdf", "page": 1, ...}
            }
        ]
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        
        # Chroma requires flat metadata values (str, int, float, bool)
        metadatas = []
        for c in chunks:
            raw_meta = c.get("metadata", {})
            flat_meta = {}
            for k, v in raw_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[k] = v
                else:
                    flat_meta[k] = str(v)
            metadatas.append(flat_meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the collection for the closest chunks to the query string.
        Returns top_k results with text, metadata, and relevance similarity scores.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [1.0] * len(docs)
        ids = results["ids"][0] if results["ids"] else [""] * len(docs)

        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance (0 to 2). Convert to Cosine Similarity score: 1 - distance
            similarity_score = max(0.0, 1.0 - dist)
            formatted_results.append({
                "chunk_id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "score": round(similarity_score, 4)
            })

        return formatted_results

    def clear(self):
        """Clears all documents in the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
