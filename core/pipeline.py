from typing import List, Dict, Any, Optional
import json

from core.query.router import IntentRouter, QueryIntent
from core.query.rewriter import QueryRewriter
from core.indexing.dense_index import DenseVectorStore
from core.indexing.sparse_index import SparseBM25Index
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.reranker import CrossEncoderReranker
from core.generation.llm_client import LLMClient
from core.generation.generator import GroundedGenerator
from core.generation.guardrails import HallucinationGuardrail

class SmartDocsPipeline:
    """
    Master Production Orchestrator:
    Combines Routing -> Rewriting -> Hybrid Search -> Reranking -> Grounded Generation -> Guardrails.
    """

    def __init__(
        self,
        dense_store: Optional[DenseVectorStore] = None,
        sparse_index: Optional[SparseBM25Index] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.dense_store = dense_store or DenseVectorStore()
        self.sparse_index = sparse_index or SparseBM25Index()
        self.llm_client = llm_client or LLMClient()

        # Wire all core components
        self.router = IntentRouter()
        self.rewriter = QueryRewriter(llm_client=self.llm_client)
        self.hybrid_retriever = HybridRetriever(
            dense_store=self.dense_store,
            sparse_index=self.sparse_index,
            rrf_k=60
        )
        self.reranker = CrossEncoderReranker(relevance_threshold=0.35)
        self.generator = GroundedGenerator(llm_client=self.llm_client)
        self.guardrail = HallucinationGuardrail(llm_client=self.llm_client)

    def answer_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes the complete multi-strategy RAG pipeline and returns
        the answer with full observability traces.
        """
        trace: Dict[str, Any] = {
            "original_query": query,
            "intent": None,
            "rewritten_query": None,
            "retrieved_count": 0,
            "reranked_count": 0,
            "chunks_used": [],
            "citations": [],
            "guardrail_verified": True
        }

        # Step 1: Intent Routing (0ms filter)
        route_decision = self.router.route(query)
        trace["intent"] = route_decision["intent"].value

        if not route_decision["should_retrieve"]:
            return {
                "answer": route_decision["direct_response"],
                "citations": [],
                "grounded": True,
                "trace": trace
            }

        # Step 2: Contextual Query Rewriting
        rewritten_q = self.rewriter.rewrite_with_history(query, chat_history)
        trace["rewritten_query"] = rewritten_q

        # Step 3: Multi-Strategy Hybrid Retrieval (ChromaDB + BM25 + RRF)
        raw_candidates = self.hybrid_retriever.search(rewritten_q, top_k=top_k * 3)
        trace["retrieved_count"] = len(raw_candidates)

        # Step 4: Cross-Encoder Reranking & Quality Threshold Filtering
        reranked_chunks = self.reranker.rerank(rewritten_q, raw_candidates, top_n=top_k)
        trace["reranked_count"] = len(reranked_chunks)

        if not reranked_chunks:
            # Out-of-domain short circuit
            return {
                "answer": "I don't know based on the provided documents. Your uploaded files do not contain information about this question.",
                "citations": [],
                "grounded": True,
                "trace": trace
            }

        trace["chunks_used"] = [
            {
                "chunk_id": c["chunk_id"],
                "score": c.get("rerank_score", 0.0),
                "source": c.get("metadata", {}).get("source", "unknown"),
                "page": c.get("metadata", {}).get("page", 1)
            }
            for c in reranked_chunks
        ]

        # Step 5: Grounded Answer Generation with Citations
        generation_output = self.generator.generate_answer(rewritten_q, reranked_chunks)
        answer_text = generation_output["answer"]
        citations = generation_output["citations"]
        trace["citations"] = citations

        # Step 6: Hallucination Guardrail Verification
        is_grounded = self.guardrail.verify(answer_text, reranked_chunks)
        trace["guardrail_verified"] = is_grounded

        if not is_grounded:
            answer_text = "I don't know based on the provided documents (Flagged by Hallucination Guardrail)."

        return {
            "answer": answer_text,
            "citations": citations,
            "grounded": is_grounded,
            "trace": trace
        }
