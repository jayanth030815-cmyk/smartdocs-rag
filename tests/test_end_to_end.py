import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.pipeline import SmartDocsPipeline
from core.indexing.dense_index import DenseVectorStore
from core.indexing.sparse_index import SparseBM25Index

def test_full_pipeline():
    print("\n" + "=" * 75)
    print("🚀 SMARTDOCS: END-TO-END MASTER PIPELINE TEST")
    print("=" * 75)

    # 1. Initialize Pipeline
    dense_store = DenseVectorStore(collection_name="test_e2e_collection")
    dense_store.clear()
    sparse_index = SparseBM25Index()

    # 2. Add realistic corporate policy chunks
    sample_policy_chunks = [
        {
            "chunk_id": "chunk_hr_1",
            "text": "SmartDocs Annual Leave Policy: Full-time employees receive 25 days of paid annual vacation per calendar year.",
            "metadata": {"source": "hr_policy.pdf", "page": 4}
        },
        {
            "chunk_id": "chunk_sec_1",
            "text": "DevOps Error Handling: When encountering ERR-504-GATEWAY on the production cluster, execute 'docker restart nginx_proxy'.",
            "metadata": {"source": "devops_runbook.pdf", "page": 12}
        },
        {
            "chunk_id": "chunk_finance_1",
            "text": "Financial Reimbursement: Travel expenses must be submitted within 14 days of trip completion with attached receipts.",
            "metadata": {"source": "finance_guidelines.pdf", "page": 2}
        }
    ]

    print("📥 1. Ingesting test documents into Dense (ChromaDB) + Sparse (BM25)...")
    dense_store.add_chunks(sample_policy_chunks)
    sparse_index.add_chunks(sample_policy_chunks)
    print("   -> Successfully indexed!")

    pipeline = SmartDocsPipeline(dense_store=dense_store, sparse_index=sparse_index)

    # TEST CASE 1: Chit-Chat Routing (Bypasses Search)
    print("\n[TEST 1: Chit-Chat Fast Routing]")
    q1 = "Good morning! How are you?"
    res1 = pipeline.answer_query(q1)
    print(f"   ❓ User: \"{q1}\"")
    print(f"   🏷️  Intent: {res1['trace']['intent']}")
    print(f"   💬 Reply : \"{res1['answer']}\"")
    print(f"   🔍 DB Searches Performed: {res1['trace']['retrieved_count']}")
    assert res1["trace"]["intent"] == "chitchat", "Expected chitchat intent!"
    assert res1["trace"]["retrieved_count"] == 0, "Chitchat should NOT search DB!"

    # TEST CASE 2: Multi-Strategy Retrieval + Exact Code Matching
    print("\n[TEST 2: Technical Document Query with Error Code]")
    q2 = "How to resolve ERR-504-GATEWAY error?"
    res2 = pipeline.answer_query(q2)
    print(f"   ❓ User: \"{q2}\"")
    print(f"   🏷️  Intent: {res2['trace']['intent']}")
    print(f"   💬 Answer: \"{res2['answer']}\"")
    print(f"   🏷️  Citations: {res2['citations']}")
    print(f"   🛡️  Guardrail Verified: {res2['trace']['guardrail_verified']}")
    assert len(res2["citations"]) > 0, "Expected at least 1 citation!"

    # TEST CASE 3: Multi-turn Pronoun Resolution & Disambiguation
    print("\n[TEST 3: Multi-Turn Conversation Disambiguation]")
    history = [
        {"role": "user", "content": "Tell me about the annual vacation policy."},
        {"role": "assistant", "content": "Full-time employees receive 25 days of vacation [Source: hr_policy.pdf, Page: 4]."}
    ]
    q3 = "How many days is it per year?"
    res3 = pipeline.answer_query(q3, chat_history=history)
    print(f"   ❓ Vague Follow-up: \"{q3}\"")
    print(f"   ✨ Rewritten Query: \"{res3['trace']['rewritten_query']}\"")
    print(f"   💬 Answer: \"{res3['answer']}\"")

    # TEST CASE 4: Out-of-Domain / Fallback Verification
    print("\n[TEST 4: Out-of-Domain Question]")
    q4 = "How do I make chocolate chip cookies?"
    res4 = pipeline.answer_query(q4)
    print(f"   ❓ User: \"{q4}\"")
    print(f"   💬 Answer: \"{res4['answer']}\"")
    assert "i don't know" in res4["answer"].lower()

    print("\n" + "=" * 75)
    print("🎉 ALL END-TO-END PIPELINE TESTS PASSED 100%!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    test_full_pipeline()
