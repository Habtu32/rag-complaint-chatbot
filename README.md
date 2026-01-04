# Intelligent Complaint Analysis for Financial Services

This repository contains the interim work for **Week 7: Intelligent Complaint Analysis** as part of the 10 Academy Artificial Intelligence Mastery program. The project aims to build a **Retrieval-Augmented Generation (RAG) chatbot** that transforms customer complaints into actionable insights for internal teams at CrediTrust Financial.

---

## **Project Overview**

CrediTrust Financial receives thousands of consumer complaints per month across multiple financial products. Currently, internal teams manually review complaints, which is time-consuming and inconsistent.

The goal of this project is to enable **fast, evidence-backed insights** by combining semantic search with language models. Users can query complaint data in plain English and receive concise, relevant answers.

**Target Product Categories:**

* Credit Cards
* Personal Loans
* Buy Now, Pay Later (BNPL)
* Savings Accounts
* Money Transfers

---

## **Repository Structure**

```
rag-complaint-chatbot/
├── data/
│   ├── raw/                        # Original CFPB dataset
│   └── processed/                  # Cleaned CSV for Task 2
├── vector_store/                   # Persisted FAISS/ChromaDB index (not uploaded)
├── notebooks/
│   ├── task1_eda.ipynb             # EDA and preprocessing
│   └── task2_embedding.ipynb       # Text chunking and embeddings
├── src/
│   └── build_vector_store.py       # Reproducible vector store script
├── app.py                          # Placeholder for future Gradio UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## **Interim Tasks Completed**

### **Task 1 – EDA and Preprocessing**

* Explored the full CFPB complaint dataset (~9.6 million records)
* Filtered for missing narratives (~69% missing removed)
* Focused on the 5 target product categories
* Analyzed complaint lengths and distributions
* Cleaned and saved the filtered dataset:

  ```
  data/processed/task1_filtered.csv
  ```

### **Task 2 – Text Chunking, Embedding, and Vector Store**

* Created a stratified sample (12,000 complaints) for embedding
* Implemented **chunking** (500 chars with 100 char overlap) to improve semantic accuracy
* Generated vector embeddings using **sentence-transformers/all-MiniLM-L6-v2**
* Built a **FAISS vector store** with metadata for retrieval
* Scripted the process in `src/build_vector_store.py` to allow reproducibility

> ⚠️ The vector store folder is **not uploaded** due to size. Use the script to rebuild locally.

---

## **Setup Instructions**

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

2. **Create a Python virtual environment:**

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run Task 1 notebook (EDA & preprocessing)**
   Open `notebooks/task1_eda.ipynb` in Jupyter or VSCode.

5. **Build the vector store for Task 2:**

```bash
python src/build_vector_store.py
```

This script will generate embeddings and save the vector store locally under `vector_store/`.

---

## **Future Work (Next Steps)**

* **Task 3 – RAG Core Logic:**

  * Implement semantic retrieval using the full vector store
  * Design LLM prompt templates and generate answers
  * Evaluate responses qualitatively

* **Task 4 – Interactive UI:**

  * Build a Gradio chat interface
  * Display retrieved source chunks for transparency and trust

---

## **Repository Best Practices**

* `.gitignore` excludes:

  * `vector_store/`
  * Temporary or intermediate files

* Code is modular with clear functions and docstrings

* Notebooks include Markdown explanations for clarity

---

## **References**

* [Gradio Docs](https://www.gradio.app/docs)
* [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki/Getting-started)
* [ChromaDB Docs](https://docs.trychroma.com/getting-started)
* [Sentence Transformers](https://www.sbert.net/docs/quickstart.html)
* [RAG with Hugging Face](https://huggingface.co/blog/rag)

