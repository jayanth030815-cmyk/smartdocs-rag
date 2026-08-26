from typing import Dict, Any, List
from core.generation.llm_client import LLMClient

class HallucinationGuardrail:
    """
    Verifies that the generated answer is strictly entailed by the context chunks.
    Acts as a quality referee before presenting the answer to the user.
    """

    VERIFICATION_PROMPT = """You are a strict Fact-Checking Judge.

CONTEXT EXCERPTS:
{context}

GENERATED ANSWER TO VERIFY:
{answer}

TASK:
Check if EVERY single factual claim in the GENERATED ANSWER is 100% supported by the CONTEXT.
If the answer makes up outside facts or contradicts the context, return: UNGROUNDED.
If the answer is fully supported, return: GROUNDED.

OUTPUT (ONLY return GROUNDED or UNGROUNDED):"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def verify(self, answer: str, chunks: List[Dict[str, Any]]) -> bool:
        """Returns True if the answer is grounded, False if hallucinated."""
        # If the model honestly refused ("I don't know"), it's not a hallucination
        if "i don't know" in answer.lower():
            return True

        if not chunks:
            return False

        context_text = "\n".join([c["text"] for c in chunks])
        prompt = self.VERIFICATION_PROMPT.format(context=context_text, answer=answer)

        decision = self.llm_client.generate(prompt)
        return "grounded" in decision.lower() and "ungrounded" not in decision.lower()
