import os
import re
from typing import List, Dict, Any, Optional

class QueryRewriter:
    """
    Transforms user queries to maximize retrieval recall:
    1. Contextual Rewriting: Resolves pronouns using chat history.
    2. Multi-Query Expansion: Generates 3 alternative search perspectives.
    3. Query Decomposition: Breaks compound questions into sub-queries.
    4. Step-Back Prompting: Generates a higher-level, broader conceptual query.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def rewrite_with_history(
        self,
        current_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Resolves pronouns (it, that, they) using previous conversation context."""
        if not chat_history:
            return current_query

        pronoun_pattern = re.compile(
            r"\b(it|its|that|this|they|them|these|those|the former|the latter|the first one|the second one|he|she|their|what about)\b",
            re.IGNORECASE
        )

        has_pronoun = bool(pronoun_pattern.search(current_query))
        
        if self.llm_client and has_pronoun:
            return self._llm_rewrite(current_query, chat_history)

        if has_pronoun and len(chat_history) >= 2:
            last_user_msg = chat_history[-2]["content"] if chat_history[-2]["role"] == "user" else chat_history[-1]["content"]
            words = [w for w in last_user_msg.split() if len(w) > 3 and w.lower() not in ["what", "when", "where", "how", "tell"]]
            if words:
                subject_hint = " ".join(words[:3])
                return f"{current_query} (regarding {subject_hint})"

        return current_query

    def generate_multi_queries(self, query: str, num_variations: int = 3) -> List[str]:
        """Generates alternative search angles to capture different vocabulary."""
        variations = [query]

        if self.llm_client:
            llm_variations = self._llm_expand_queries(query, num_variations - 1)
            variations.extend(llm_variations)
            return list(dict.fromkeys(variations))[:num_variations]

        clean_q = re.sub(r'^(what is|how to|can i|explain|tell me about)\s+', '', query, flags=re.IGNORECASE).strip()
        if clean_q and clean_q != query:
            variations.append(f"{clean_q} overview guide")
            variations.append(f"{clean_q} details and policy")

        return list(dict.fromkeys(variations))[:num_variations]

    def decompose_compound_query(self, query: str) -> List[str]:
        """Splits compound questions ('and', 'also') into atomic sub-queries."""
        parts = re.split(r'\b(?:and also|and|as well as|\;)\b', query, flags=re.IGNORECASE)
        sub_queries = [p.strip() for p in parts if len(p.strip()) > 5]
        return sub_queries if len(sub_queries) > 1 else [query]

    def generate_step_back_query(self, query: str) -> str:
        """
        Step-Back Prompting:
        Takes a specific question and abstracts it to a broader, high-level concept query
        to retrieve foundational principles.
        """
        if self.llm_client:
            prompt = (
                f"You are an expert at Step-Back Prompting. Given this specific query: '{query}', "
                "write a broader, higher-level conceptual search query that retrieves the core principles behind it. "
                "Output ONLY the step-back query."
            )
            try:
                return self.llm_client.generate(prompt).strip()
            except Exception:
                pass

        # Heuristic Step-Back: Strip specifics (numbers, error codes, page numbers) to get broad concept
        broad_q = re.sub(r'\b(on page \d+|in chapter \d+|error code \w+|line \d+|\#\d+)\b', '', query, flags=re.IGNORECASE).strip()
        return f"Core principles and general concepts of {broad_q}"

    def _llm_rewrite(self, query: str, history: List[Dict[str, str]]) -> str:
        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a completely standalone search query. "
            f"Chat History:\n{history}\n\n"
            f"Follow-up Question: {query}\n\n"
            "Standalone Search Query:"
        )
        try:
            return self.llm_client.generate(prompt).strip()
        except Exception:
            return query

    def _llm_expand_queries(self, query: str, count: int) -> List[str]:
        prompt = (
            f"Generate {count} distinct search queries expressing the same intent as: '{query}'. "
            "One per line."
        )
        try:
            response = self.llm_client.generate(prompt)
            return [line.strip() for line in response.split("\n") if line.strip()][:count]
        except Exception:
            return []
