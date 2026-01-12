# ==============================
# app.py
# Interactive RAG Chat Interface
# ==============================

import gradio as gr
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# 1. Load models
# ==============================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# ==============================
# 2. Load FAISS vector store
# ==============================

index = faiss.read_index(
    "vector_store/faiss_complaints/index.faiss"
)

with open("vector_store/faiss_complaints/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

# ==============================
# 3. Retrieval function
# ==============================

def retrieve_chunks(query, k=5):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k)

    retrieved = []
    for idx in indices[0]:
        if idx in index_to_docstore_id:
            doc_id = index_to_docstore_id[idx]
            doc = docstore._dict.get(doc_id)
            if doc:
                retrieved.append(doc)

    return retrieved

# ==============================
# 4. RAG answer generation
# ==============================

def answer_question(question):
    docs = retrieve_chunks(question, k=5)

    # Strong hallucination guard
    if len(docs) < 2:
        return (
            "I do not have enough information to answer this question.",
            "Insufficient relevant complaint excerpts retrieved."
        )

    context = "\n\n".join(
        [f"- {d.page_content}" for d in docs[:3]]
    )

    prompt = f"""
You are a financial analyst assistant for CrediTrust.

RULES:
- Use ONLY the complaint excerpts below.
- Do NOT use outside knowledge.
- If the answer is not clearly supported, say:
  "I do not have enough information to answer this question."

Complaint Excerpts:
{context}

Question:
{question}

Answer (concise, analytical):
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,   # 🔑 CRITICAL FIX
            num_beams=4        # Deterministic reasoning
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    sources = "\n\n".join(
        [f"Source {i+1}:\n{d.page_content}" for i, d in enumerate(docs[:3])]
    )

    return answer.strip(), sources

# ==============================
# 5. Gradio UI
# ==============================

with gr.Blocks(title="CrediTrust Complaint Analysis Assistant") as demo:

    gr.Markdown("## CrediTrust Consumer Complaint Assistant")
    gr.Markdown(
        "Ask questions about customer complaints. "
        "All answers are grounded in retrieved complaint narratives."
    )

    question_input = gr.Textbox(
        label="Your question",
        placeholder="What issues do customers report with credit cards?",
        lines=2
    )

    ask_btn = gr.Button("Ask")
    clear_btn = gr.Button("Clear")

    answer_output = gr.Textbox(
        label="AI Answer",
        lines=6
    )

    sources_output = gr.Textbox(
        label="Sources",
        lines=10
    )

    ask_btn.click(
        answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    clear_btn.click(
        lambda: ("", ""),
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()