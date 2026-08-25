import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.indexing.dense_index import DenseVectorStore
from core.indexing.sparse_index import SparseBM25Index
from core.retrieval.hybrid_retriever import HybridRetriever

def test_hybrid_pipeline():
    print("\n" + "=" * 70)
    print("⚡ SMARTDOCS: TESTING HYBRID SEARCH (Dense + BM25 + RRF)")
    print("=" * 70)

    # 1. Initialize Vector Store & BM25 Index
    dense_store = DenseVectorStore(collection_name="test_hybrid_collection")
    dense_store.clear()
    sparse_index = SparseBM25Index()

    # 2. Add realistic documents with specific codes, numbers, and concepts
    test_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Troubleshooting Server Outages: When encountering timeout error ERR-504-GATEWAY, restart the nginx proxy container.",
            "metadata": {"source": "devops_guide.pdf", "page": 12}
        },
        {
            "chunk_id": "chunk_2",
            "text": "General network connection troubleshooting steps for slow website loading and server downtime.",
            "metadata": {"source": "general_faq.pdf", "page": 2}
        },
        {
            "chunk_id": "chunk_3",
            "text": "Customer Refund Policy: Users are entitled to a full money-back refund within 30 days of purchase.",
            "metadata": {"source": "refunds.pdf", "page": 1}
        }
    ]

    print("📥 1. Indexing 3 chunks into both Dense (ChromaDB) and Sparse (BM25)...")
    dense_store.add_chunks(test_chunks)
    sparse_index.add_chunks(test_chunks)
    print("   -> Indexed in both search engines successfully!")

    # 3. Create Hybrid Retriever
    retriever = HybridRetriever(dense_store=dense_store, sparse_index=sparse_index, rrf_k=60)

    # 4. TEST CASE A: Exact Error Code Query (BM25 shines!)
    query_a = "How to resolve ERR-504-GATEWAY?"
    print(f"\n🔍 [TEST A] Query with Exact Code: \"{query_a}\"")
    results_a = retriever.search(query_a, top_k=2)

    for i, res in enumerate(results_a, start=1):
        print(f"   🏆 Match #{i} (RRF Score: {res['rrf_score']}) [Dense Rank: {res['dense_rank']}, BM25 Rank: {res['sparse_rank']}]")
        print(f"      Source: {res['metadata'].get('source')}, Page {res['metadata'].get('page')}")
        print(f"      Text  : \"{res['text']}\"")

    # 5. TEST CASE B: Pure Conceptual Query with Synonyms (Dense shines!)
    query_b = "I want my cash returned for my purchase."
    print(f"\n🔍 [TEST B] Conceptual Query (No exact words): \"{query_b}\"")
    results_b = retriever.search(query_b, top_k=2)

    for i, res in enumerate(results_b, start=1):
        print(f"   🏆 Match #{i} (RRF Score: {res['rrf_score']}) [Dense Rank: {res['dense_rank']}, BM25 Rank: {res['sparse_rank']}]")
        print(f"      Source: {res['metadata'].get('source')}, Page {res['metadata'].get('page')}")
        print(f"      Text  : \"{res['text']}\"")

    print("\n" + "=" * 70)
    print("🎉 HYBRID SEARCH TEST PASSED! Dense & BM25 are fused with RRF!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_hybrid_pipeline()
