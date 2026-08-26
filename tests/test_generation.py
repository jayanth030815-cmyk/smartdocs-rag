import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from core.generation.llm_client import LLMClient
from core.generation.generator import GroundedGenerator
from core.generation.guardrails import HallucinationGuardrail

def test_generation_and_guardrails():
    print("\n" + "=" * 70)
    print("👑 SMARTDOCS: TESTING GROUNDED GENERATION, CITATIONS & GUARDRAILS")
    print("=" * 70)

    llm = LLMClient()
    generator = GroundedGenerator(llm_client=llm)
    guardrail = HallucinationGuardrail(llm_client=llm)

    sample_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "The refund policy allows customers to return unopened items within 30 days with receipt.",
            "metadata": {"source": "return_policy.pdf", "page": 2}
        }
    ]

    # TEST CASE 1: Grounded Answer with Citations
    q1 = "What is the return window?"
    print(f"\n[TEST 1: Grounded Answer Synthesis]")
    print(f"   ❓ Question: \"{q1}\"")
    result1 = generator.generate_answer(q1, sample_chunks)

    print(f"   💬 Generated Answer : \"{result1['answer']}\"")
    print(f"   🏷️  Citations List  : {result1['citations']}")
    print(f"   🛡️  Grounded Status  : {result1['grounded']}")

    # TEST CASE 2: Hallucination Guardrail Check
    print(f"\n[TEST 2: Hallucination Guardrail Check]")
    is_valid = guardrail.verify(result1["answer"], sample_chunks)
    print(f"   🛡️  Guardrail Verified Grounded: {is_valid}")
    assert is_valid, "Expected answer to pass guardrail check!"

    # TEST CASE 3: Out-of-Domain Fallback ("I don't know")
    print(f"\n[TEST 3: Out-of-Domain / Empty Chunks Fallback]")
    result3 = generator.generate_answer("How do I bake a cheesecake?", [])
    print(f"   💬 Fallback Answer: \"{result3['answer']}\"")
    assert "i don't know" in result3["answer"].lower(), "Expected polite refusal for empty chunks!"

    print("\n" + "=" * 70)
    print("🎉 ALL GENERATION, CITATION & GUARDRAIL TESTS PASSED!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_generation_and_guardrails()
