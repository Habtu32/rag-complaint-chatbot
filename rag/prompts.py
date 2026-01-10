# prompts.py
from string import Template
from .config import PROMPT_TEMPLATE, MAX_CONTEXT_LENGTH

def build_rag_prompt(question: str, context: str) -> str:
    """Safely format prompt with length guard"""
    context_safe = context[:MAX_CONTEXT_LENGTH]
    if len(context_safe.strip()) < 150:
        context_safe = "[Insufficient relevant context retrieved]"

    return PROMPT_TEMPLATE.format(
        context=context_safe,
        question=question
    )