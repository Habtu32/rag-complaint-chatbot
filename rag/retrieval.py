# retrieval.py
from langchain_core.documents import Document
from typing import List, Tuple

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Simple, reliable similarity search"""
        return self.vector_store.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """If you need scores (some vectorstores support it)"""
        # FAISS by default doesn't return scores in similarity_search
        # Use similarity_search_with_score when available
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except AttributeError:
            docs = self.retrieve(query, k)
            return [(doc, 0.0) for doc in docs]  # fallback

    def get_context_string(self, docs: List[Document], max_total_chars: int = 3000) -> str:
        parts = []
        total = 0

        for doc in docs:
            text = getattr(doc, 'page_content', str(doc)).strip()
            if not text:
                continue

            if total + len(text) > max_total_chars:
                break

            parts.append(text)
            total += len(text)

        return "\n\n".join(parts)