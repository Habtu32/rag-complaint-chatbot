"""Build a FAISS vector store from complaint narratives.

Creates chunked documents with metadata and saves a FAISS index for downstream retrieval.

Usage example:
    python src/build_vector_store.py --data ../data/processed/filtered_complaints.csv \
        --sample-size 12000 --chunk-size 500 --chunk-overlap 100 --output-dir ../vector_store/faiss_complaints
"""
from pathlib import Path
import argparse
import logging
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


LOGGER = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    LOGGER.info("Loaded %d rows from %s", len(df), path)
    return df


def stratified_sample(df: pd.DataFrame, sample_size: int, stratify_col: str = "product_category") -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        LOGGER.info("No downsampling requested or sample_size >= full dataset; using full dataset (%d rows)", len(df))
        return df.copy()

    try:
        df_sampled, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[stratify_col],
            random_state=42,
        )
        LOGGER.info("Stratified sample: %d rows", len(df_sampled))
        return df_sampled
    except Exception as exc:
        LOGGER.warning("Stratified sampling failed (%s); falling back to simple random sample", exc)
        return df.sample(n=sample_size, random_state=42)


def create_documents(
    df: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 100
) -> List[Document]:
    text_col = "Consumer complaint narrative"
    id_col = "Complaint ID"
    cat_col = "product_category"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs: List[Document] = []
    skipped = 0

    for _, row in df.iterrows():
        text = row.get(text_col)
        if pd.isna(text) or not str(text).strip():
            skipped += 1
            continue

        chunks = text_splitter.split_text(str(text))
        for chunk in chunks:
            source = f"This chunk came from complaint {row[id_col]}, product {row[cat_col]}."
            metadata = {
                "complaint_id": row[id_col],
                "product_category": row[cat_col],
                "source": source,
            }
            docs.append(Document(page_content=chunk, metadata=metadata))

    LOGGER.info("Created %d chunk documents (skipped %d rows with no narrative)", len(docs), skipped)
    return docs


def build_faiss_index(docs: List[Document], model_name: str) -> FAISS:
    LOGGER.info("Initializing embedding model: %s", model_name)
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    LOGGER.info("Building FAISS index from %d documents...", len(docs))
    vector_store = FAISS.from_documents(docs, embedding_model)
    LOGGER.info("FAISS index built")
    return vector_store


def save_vector_store(vector_store: FAISS, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving vector store to %s", output_dir)
    vector_store.save_local(str(output_dir))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS vector store from complaint narratives")
    p.add_argument("--data", type=Path, default=Path("../data/processed/filtered_complaints.csv"))
    p.add_argument("--sample-size", type=int, default=12000)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--chunk-overlap", type=int, default=100)
    p.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--output-dir", type=Path, default=Path("../vector_store/faiss_complaints"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    data_path = args.data
    if not data_path.exists():
        LOGGER.error("Data file not found: %s", data_path)
        raise SystemExit(1)

    df = load_data(data_path)
    df_sampled = stratified_sample(df, args.sample_size)

    docs = create_documents(df_sampled, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if not docs:
        LOGGER.error("No documents were created; exiting")
        raise SystemExit(1)

    vector_store = build_faiss_index(docs, model_name=args.model_name)
    save_vector_store(vector_store, args.output_dir)

    LOGGER.info("Done. Saved vector store at %s", args.output_dir)


if __name__ == "__main__":
    main()
