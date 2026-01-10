# run_example.py
from your_llm_loader import load_llm  # ← your own LLM loader
from vectorstore import load_or_create_vectorstore  # your function
from rag_pipeline import RAGPipeline
from config import DEFAULT_K

# Evaluation questions
EVAL_QUESTIONS = [
    "What are the most common issues users report with credit cards?",
    "Why are customers unhappy with Buy Now, Pay Later services?",
    "What complaints are frequently raised about money transfers?",
    "Are there recurring issues related to account closures?",
    "What customer pain points suggest compliance or fraud risks?"
]

def main():
    print("Loading LLM...")
    llm = load_llm()

    print("Loading/creating vector store...")
    vector_store = load_or_create_vectorstore()

    pipeline = RAGPipeline(vector_store, llm)

    print("\n" + "="*70 + "\nRunning evaluation questions\n" + "="*70)

    for q in EVAL_QUESTIONS:
        print(f"\nQUESTION: {q}")
        result = pipeline.answer(q, k=5, return_timings=True, return_docs=True)

        print("ANSWER:")
        print(result["answer"][:500], "..." if len(result["answer"]) > 500 else "")
        print("\nTimings:", result["timings"])
        if result.get("sources"):
            print("Top source excerpt:")
            print(result["sources"][0].page_content[:280] + "...")
        print("-"*80)

if __name__ == "__main__":
    main()