import re
from enum import Enum
from typing import Dict, Any, Optional

class QueryIntent(str, Enum):
    CHITCHAT = "chitchat"
    DOCUMENT_QUERY = "document_query"
    META_QUERY = "meta_query"

class IntentRouter:
    """
    Classifies incoming user messages into:
    1. CHITCHAT: Greetings, pleasantries, small talk, identity questions.
    2. DOCUMENT_QUERY: Questions requiring factual document retrieval.
    3. META_QUERY: Questions about loaded files/catalogs.
    """

    # Comprehensive fast-path regex patterns for chit-chat & pleasantries
    CHITCHAT_PATTERNS = [
        r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day))\b",
        r"^(how\s+are\s+you|how\'?s\s+it\s+going|what\'?s\s+up|how\s+do\s+you\s+do)\b",
        r"^(who\s+(are\s+you|created\s+you|made\s+you|built\s+you|developed\s+you))\b",
        r"^(what\s+is\s+your\s+name|what\s+can\s+you\s+do|what\s+are\s+you|help)\b",
        r"^(thank\s+you|thanks|thank\s+you\s+so\s+much|thx)\b",
        r"^(bye|goodbye|see\s+you|have\s+a\s+(good|great)\s+day)\b",
        r"^(tell\s+me\s+a\s+joke|are\s+you\s+(human|an?\s+ai|a\s+robot)|nice\s+to\s+meet\s+you)\b"
    ]

    # Patterns indicating questions about the uploaded files themselves
    META_PATTERNS = [
        r"^(what\s+(documents|files|pdfs)|list\s+(documents|files)|show\s+uploaded|what\s+is\s+loaded)\b",
        r"^(summarize\s+(the\s+document|all\s+files|the\s+pdf)|give\s+me\s+an\s+overview)\b"
    ]

    def __init__(self):
        self.compiled_chitchat = [re.compile(p, re.IGNORECASE) for p in self.CHITCHAT_PATTERNS]
        self.compiled_meta = [re.compile(p, re.IGNORECASE) for p in self.META_PATTERNS]

    def route(self, query: str) -> Dict[str, Any]:
        """
        Routes the user query to the appropriate intent and provides a direct response if chitchat.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "intent": QueryIntent.CHITCHAT,
                "direct_response": "Hello! How can I help you with your documents today?",
                "should_retrieve": False
            }

        # 1. Check Fast-Path Chit-Chat Rules
        for pattern in self.compiled_chitchat:
            if pattern.search(clean_query):
                return {
                    "intent": QueryIntent.CHITCHAT,
                    "direct_response": self._generate_chitchat_response(clean_query),
                    "should_retrieve": False
                }

        # 2. Check Meta Questions (about uploaded documents)
        for pattern in self.compiled_meta:
            if pattern.search(clean_query):
                return {
                    "intent": QueryIntent.META_QUERY,
                    "direct_response": None,
                    "should_retrieve": True
                }

        # 3. Default to Document Query (requires retrieval)
        return {
            "intent": QueryIntent.DOCUMENT_QUERY,
            "direct_response": None,
            "should_retrieve": True
        }

    @staticmethod
    def _generate_chitchat_response(query: str) -> str:
        """Generates friendly, helpful conversational responses."""
        q = query.lower()
        if any(w in q for w in ["hi", "hello", "hey", "morning", "evening", "greetings"]):
            return "Hello! I am SmartDocs AI. Upload your documents or ask me any question about them!"
        elif any(w in q for w in ["who are you", "what is your name", "who created you", "who made you", "who built you"]):
            return "I am SmartDocs, an advanced Multi-Strategy RAG assistant designed to answer your document questions factually."
        elif any(w in q for w in ["thank", "thanks"]):
            return "You're very welcome! Let me know if you have any more questions about your documents."
        elif any(w in q for w in ["how are you", "how's it going"]):
            return "I'm doing great and ready to help you analyze your documents! What would you like to know?"
        elif any(w in q for w in ["bye", "goodbye"]):
            return "Goodbye! Have a great day ahead."
        else:
            return "Hello! I'm here to help you search and analyze your uploaded documents."
