import os
import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG / DEFINITIONS
# ==============================
MODEL_NAME = "BAAI/bge-base-en-v1.5"
DATA_DIR = "data"                       # Folder for outputs
DEFAULT_CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
DEFAULT_INDEX_FILE = os.path.join(DATA_DIR, "thesis.index")
TOP_K = 5  # For retrieval later (optional)


# ==============================
# HELPER FUNCTIONS
# ==============================
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def load_chunks(path=DEFAULT_CHUNKS_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts, model):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )


def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalized vectors
    index.add(embeddings)
    return index


def save_index(index, chunks, path=DEFAULT_INDEX_FILE):
    faiss.write_index(index, path)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)


# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Embed chunks and create FAISS vector store")
    parser.add_argument("--chunks_file", default=DEFAULT_CHUNKS_FILE, help="Path to chunks.json")
    parser.add_argument("--output", default=DEFAULT_INDEX_FILE, help="Output FAISS index path")
    args = parser.parse_args()

    # Ensure output folder exists
    ensure_data_dir()

    print("Loading chunks...")
    chunks = load_chunks(args.chunks_file)
    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = embed_texts(texts, model)

    print("Building FAISS index...")
    index = build_index(embeddings)

    print(f"Saving FAISS index to {args.output} ...")
    save_index(index, chunks, args.output)

    print(f"Stored {len(chunks)} vectors.")
    print("Done.")


if __name__ == "__main__":
    main()

