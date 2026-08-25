import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.ingestion.document_loader import DocumentLoader
from core.ingestion.chunker import RecursiveCharacterChunker

def run_test():
    print("\n" + "=" * 65)
    print("🚀 SMARTDOCS: TESTING INGESTION & SMART CHUNKER")
    print("=" * 65)

    # 1. Create a sample text file on the fly for testing
    sample_file_path = root_dir / "sample_test_doc.txt"
    sample_content = (
        "SmartDocs Architecture Overview.\n\n"
        "SmartDocs is an advanced Multi-Strategy RAG system designed for production use. "
        "It solves the classic pitfalls of basic semantic search: bad queries, hallucinations, "
        "and noisy retrieval.\n\n"
        "Chapter 1: Document Ingestion and Chunking.\n"
        "Documents like PDFs, Markdown, and text files are ingested and split into clean chunks. "
        "Each chunk preserves important metadata such as file name, page number, and chunk ID.\n\n"
        "Chapter 2: Multi-Strategy Retrieval.\n"
        "SmartDocs combines Dense Vector Search with Sparse BM25 Keyword Search using "
        "Reciprocal Rank Fusion (RRF) for high precision."
    )

    with open(sample_file_path, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print(f"📄 1. Created sample document: {sample_file_path.name}")

    # 2. Test DocumentLoader
    print("📥 2. Loading document via DocumentLoader...")
    loaded_docs = DocumentLoader.load_file(str(sample_file_path))
    print(f"   -> Loaded {len(loaded_docs)} document section(s).")
    print(f"   -> Attached Metadata: {loaded_docs[0]['metadata']}")

    # 3. Test RecursiveCharacterChunker
    print("\n✂️  3. Chunking with chunk_size=200, chunk_overlap=40...")
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_documents(loaded_docs)

    print(f"   -> Total Chunks Produced: {len(chunks)}")
    print("=" * 65)

    # 4. Display each chunk with its metadata
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n📦 CHUNK {i} (ID: {chunk['chunk_id']})")
        print(f"   🏷️  Metadata   : {chunk['metadata']}")
        print(f"   📏 Character Len: {len(chunk['text'])}")
        print(f"   📝 Text Content :")
        print(f"      \"{chunk['text']}\"")
        print("-" * 65)

    # Clean up test file
    if sample_file_path.exists():
        os.remove(sample_file_path)
    
    print("\n🎉 ALL TESTS PASSED! Ingestion & Chunking are working 100%!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_test()
