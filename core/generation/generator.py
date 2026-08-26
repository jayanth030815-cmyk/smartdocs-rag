from typing import List, Dict, Any
from core.generation.llm_client import LLMClient

class GroundedGenerator:
    """
    Assembles retrieved chunks into a strict context prompt and generates
    grounded answers with inline citations.
    """

    GROUNDED_PROMPT_TEMPLATE = """You are SmartDocs AI, a strictly factual document assistant.

CRITICAL INSTRUCTIONS:
1. Answer the USER QUESTION using ONLY the facts provided in the CONTEXT below.
2. Do NOT assume, extrapolate, or use outside knowledge.
3. If the context does not contain enough facts to answer, you MUST reply:
   "I don't know based on the provided documents."
4. Every factual claim MUST conclude with an inline citation matching the exact format: [Source: <filename>, Page: <page_number>].

CONTEXT:
{context}

USER QUESTION:
{question}

FACTUAL ANSWER WITH CITATIONS:"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats chunk list into clean numbered excerpts with source tags."""
        if not chunks:
            return "No relevant excerpts found."

        formatted_excerpts = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "unknown_document")
            page = meta.get("page", 1)
            cid = chunk.get("chunk_id", f"chunk_{i}")

            excerpt = f"--- [EXCERPT {i}] (Source: {source} | Page: {page} | ID: {cid}) ---\n{chunk['text'].strip()}"
            formatted_excerpts.append(excerpt)

        return "\n\n".join(formatted_excerpts)

    def generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates a cited answer and tracks source citations."""
        if not chunks:
            return {
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "grounded": False
            }

        context_text = self.format_context(chunks)
        prompt = self.GROUNDED_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question
        )

        answer_text = self.llm_client.generate(prompt)

        # Extract structured citations list from chunks
        citations = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            citations.append({
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 1),
                "chunk_id": chunk.get("chunk_id", "")
            })

        return {
            "answer": answer_text,
            "citations": citations,
            "grounded": "i don't know" not in answer_text.lower()
        }
