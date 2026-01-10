# generation.py
import time
from typing import Tuple, List, Any
from .prompts import build_rag_prompt

class Generator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt: str) -> str:
        start = time.time()
        try:
            response = self.llm(prompt)
            text = response[0]["generated_text"] if isinstance(response, list) else response
            return text.strip(), time.time() - start
        except Exception as e:
            return f"[Generation error: {str(e)}]", time.time() - start

def rag_answer_timed(
    question: str,
    retriever: 'Retriever',
    generator: 'Generator',
    k: int = 5,
    max_context_chars: int = 3000
) -> Tuple[str, List[Any], dict]:
    t0 = time.time()

    docs = retriever.retrieve(question, k=k)
    t_retrieve = time.time() - t0

    context = retriever.get_context_string(docs, max_context_chars)
    t_context = time.time() - t_retrieve - t0

    prompt = build_rag_prompt(question, context)
    answer, gen_time = generator.generate(prompt)

    timings = {
        "retrieval_s": round(t_retrieve, 2),
        "context_build_s": round(t_context, 2),
        "generation_s": round(gen_time, 2),
        "total_s": round(time.time() - t0, 2)
    }

    return answer, docs, timings