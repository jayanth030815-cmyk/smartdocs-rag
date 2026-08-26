import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Unified LLM Client supporting Google Gemini API with a fallback mock mode for offline testing.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": 0.0}  # Deterministic, grounded output!
                )
            except ImportError:
                pass

    def generate(self, prompt: str) -> str:
        """Generates a text completion from the LLM."""
        if self.client:
            try:
                response = self.client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"LLM Generation Error: {str(e)}"
        
        # Fallback Mock Mode if no API key is active yet
        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Provides realistic grounded mock responses for offline testing."""
        if "hallucination" in prompt.lower() or "check if every single" in prompt.lower():
            return "GROUNDED"
        if "step-back" in prompt.lower():
            return "Core principles and general concepts"
        if "rephrase the follow-up" in prompt.lower():
            return "Tesla Model 3 battery replacement cost"
        return "According to the provided documents, customers can return items within 30 days of purchase with a receipt [Source: return_policy.pdf, Page: 2]."
