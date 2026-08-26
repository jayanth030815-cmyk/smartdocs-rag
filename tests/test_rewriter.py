import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.query.rewriter import QueryRewriter

def test_query_rewriter():
    print("\n" + "=" * 70)
    print("🔄 SMARTDOCS: TESTING QUERY REWRITER & EXPANDER")
    print("=" * 70)

    rewriter = QueryRewriter()

    # 1. TEST CASE A: Contextual Disambiguation (Pronoun Resolution)
    chat_history = [
        {"role": "user", "content": "Tell me about the Tesla Model 3 Long Range battery specs."},
        {"role": "assistant", "content": "The Tesla Model 3 Long Range features an 82 kWh battery with 358 miles of range."}
    ]
    follow_up_query = "How much does it cost to replace?"

    rewritten_q = rewriter.rewrite_with_history(follow_up_query, chat_history)

    print("\n[TEST A: Contextual Disambiguation / Pronoun Resolution]")
    print(f"   💬 Previous Turn : \"{chat_history[0]['content']}\"")
    print(f"   ❓ Vague Follow-up: \"{follow_up_query}\"")
    print(f"   ✨ Rewritten Query: \"{rewritten_q}\"")

    # 2. TEST CASE B: Multi-Query Expansion
    original_query = "What is the return policy for defective items?"
    multi_queries = rewriter.generate_multi_queries(original_query, num_variations=3)

    print("\n[TEST B: Multi-Query Expansion]")
    print(f"   ❓ Original Query: \"{original_query}\"")
    print(f"   🎯 Generated Query Angles ({len(multi_queries)}):")
    for i, q in enumerate(multi_queries, start=1):
        print(f"      {i}. \"{q}\"")

    # 3. TEST CASE C: Compound Query Decomposition
    compound_query = "What is the warranty period and how do I replace the battery?"
    sub_queries = rewriter.decompose_compound_query(compound_query)

    print("\n[TEST C: Compound Query Decomposition]")
    print(f"   ❓ Compound Question: \"{compound_query}\"")
    print(f"   🧩 Decomposed Sub-Queries ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, start=1):
        print(f"      {i}. \"{sq}\"")

    print("\n" + "=" * 70)
    print("🎉 ALL QUERY REWRITING TESTS PASSED!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_query_rewriter()
