import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.indexing.dense_index import DenseVectorStore

def test_vector_search():
    print("\n" + "=" * 65)
    print("🧠 SMARTDOCS: TESTING DENSE VECTOR SEARCH (ChromaDB)")
    print("=" * 65)

    # 1. Initialize Vector Store
    vector_store = DenseVectorStore(collection_name="test_smartdocs")
    vector_store.clear()  # Start fresh for testing

    # 2. Add sample chunks with different topics
    sample_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "The refund policy allows customers to return unopened items within 30 days for a 100% money-back guarantee.",
            "metadata": {"source": "policy.pdf", "page": 1}
        },
        {
            "chunk_id": "chunk_2",
            "text": "Our AI system uses Reciprocal Rank Fusion to blend BM25 keyword search with dense vector embeddings.",
            "metadata": {"source": "architecture.pdf", "page": 4}
        },
        {
            "chunk_id": "chunk_3",
            "text": "To change your account password, navigate to the Settings page and click 'Security & Login'.",
            "metadata": {"source": "user_guide.pdf", "page": 7}
        }
    ]

    print("📥 1. Storing 3 sample chunks into ChromaDB...")
    vector_store.add_chunks(sample_chunks)
    print("   -> Successfully stored and indexed in vector space!")

    # 3. Test Query (Notice: We use completely DIFFERENT words than the original text!)
    test_query = "Can I get my money back if I send back the product?"
    print(f"\n❓ 2. User Query: \"{test_query}\"")
    print("   (Note: Notice we didn't use the words 'refund' or '30 days'!)")

    print("\n🔍 3. Searching ChromaDB for closest semantic matches...")
    results = vector_store.search(test_query, top_k=2)

    print("=" * 65)
    for i, res in enumerate(results, start=1):
        print(f"\n🎯 Result #{i} (Score / Similarity: {res['score']})")
        print(f"   🏷️ Source Doc : {res['metadata'].get('source')} (Page {res['metadata'].get('page')})")
        print(f"   📝 Text Content: \"{res['text']}\"")
        print("-" * 65)

    print("\n🎉 Dense Vector Search test completed successfully!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    test_vector_search()
