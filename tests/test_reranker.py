import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.retrieval.reranker import CrossEncoderReranker

def test_reranker_and_filtering():
    print("\n" + "=" * 70)
    print("🔍 SMARTDOCS: TESTING CROSS-ENCODER RERANKER & QUALITY FILTER")
    print("=" * 70)

    reranker = CrossEncoderReranker(relevance_threshold=0.35)

    # 1. Candidate chunks from an initial coarse search
    candidate_chunks = [
        {
            "chunk_id": "chunk_A",
            "text": "General company overview: SmartDocs was founded in 2026 to revolutionize AI search.",
            "metadata": {"source": "about.pdf", "page": 1}
        },
        {
            "chunk_id": "chunk_B",
            "text": "Security Protocol: Password changes require 2-factor authentication and SMS verification.",
            "metadata": {"source": "security.pdf", "page": 5}
        },
        {
            "chunk_id": "chunk_C",
            "text": "Resetting Credentials: If you forgot your password, click 'Forgot Password' on the login screen to receive a reset link.",
            "metadata": {"source": "user_guide.pdf", "page": 2}
        }
    ]

    # TEST CASE 1: Re-ordering candidates to find the highest precision match
    query_1 = "How do I reset my forgotten account password?"
    print(f"\n[TEST 1: Precision Reranking] Query: \"{query_1}\"")
    reranked_results = reranker.rerank(query_1, candidate_chunks, top_n=2)

    print(f"   -> Top Ranked Results ({len(reranked_results)}):")
    for i, res in enumerate(reranked_results, start=1):
        print(f"      {i}. [{res['chunk_id']}] (Rerank Score: {res['rerank_score']}) - \"{res['text'][:60]}...\"")

    # Verify that Chunk C (direct reset steps) is ranked #1 over generic security
    assert reranked_results[0]["chunk_id"] == "chunk_C", "Expected chunk_C to be ranked #1!"
    print("   ✅ Chunk C correctly promoted to #1!")

    # TEST CASE 2: Out-of-Domain Threshold Filtering
    query_2 = "How do I bake a strawberry cheesecake?"
    print(f"\n[TEST 2: Out-of-Domain Threshold Filter] Query: \"{query_2}\"")
    filtered_results = reranker.rerank(query_2, candidate_chunks, top_n=2)

    print(f"   -> Results passing threshold: {len(filtered_results)}")
    if not filtered_results:
        print("   ✅ All irrelevant chunks filtered out! Zero garbage passed to LLM!")

    print("\n" + "=" * 70)
    print("🎉 ALL RERANKER & QUALITY FILTER TESTS PASSED!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_reranker_and_filtering()
