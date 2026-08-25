import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.query.router import IntentRouter, QueryIntent

def test_intent_router():
    print("\n" + "=" * 65)
    print("🚦 SMARTDOCS: TESTING INTENT ROUTER")
    print("=" * 65)

    router = IntentRouter()

    test_queries = [
        ("Good morning! How are you?", QueryIntent.CHITCHAT),
        ("Who created you?", QueryIntent.CHITCHAT),
        ("What is the refund policy on Page 4?", QueryIntent.DOCUMENT_QUERY),
        ("How do I fix ERR-504 timeout error?", QueryIntent.DOCUMENT_QUERY),
        ("What documents are currently uploaded?", QueryIntent.META_QUERY),
        ("Thank you so much!", QueryIntent.CHITCHAT),
    ]

    all_passed = True
    for query, expected_intent in test_queries:
        decision = router.route(query)
        intent = decision["intent"]
        should_search = decision["should_retrieve"]
        direct_reply = decision["direct_response"]

        passed = (intent == expected_intent)
        status_icon = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False

        print(f"\n{status_icon} | Query: \"{query}\"")
        print(f"        🏷️  Expected: {expected_intent.value.upper()} | Got: {intent.value.upper()}")
        print(f"        🔍 Should Search DB : {should_search}")
        if direct_reply:
            print(f"        💬 Direct Response  : \"{direct_reply}\"")
        print("-" * 65)

    assert all_passed, "Some intent test cases failed!"
    print("\n🎉 ALL ROUTER TESTS PASSED! Chit-chat, Meta, and Doc queries are cleanly routed!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    test_intent_router()
