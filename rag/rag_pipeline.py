# rag_pipeline.py
from .retrieval import Retriever
from .generation import Generator, rag_answer_timed
from .config import DEFAULT_K

class RAGPipeline:
    def __init__(self, vector_store, llm):
        self.retriever = Retriever(vector_store)
        self.generator = Generator(llm)

    def answer(
        self,
        question: str,
        k: int = DEFAULT_K,
        return_timings: bool = False,
        return_docs: bool = False
    ):
        answer, docs, timings = rag_answer_timed(
            question=question,
            retriever=self.retriever,
            generator=self.generator,
            k=k
        )

        result = {"answer": answer}

        if return_docs:
            result["sources"] = docs
        if return_timings:
            result["timings"] = timings

        return result