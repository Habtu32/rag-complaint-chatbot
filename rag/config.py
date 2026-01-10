# config.py
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Paths
INDEX_PATH = BASE_DIR / "faiss_index"
DOCS_JSONL_PATH = BASE_DIR / "data" / "complaints_processed.jsonl"  # optional

# Retrieval
DEFAULT_K = 5
MIN_CONTEXT_LENGTH = 400      # characters
MAX_CONTEXT_LENGTH = 3200

# Prompt
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust.
Answer the question using ONLY the information explicitly stated in the complaint excerpts below.
Do NOT add general knowledge or assumptions.
Do NOT list issues not directly mentioned in the context.

If the context is insufficient, respond ONLY with:
"I do not have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""

# Model (change according to your environment)
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"   # example
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 450