import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

app = FastAPI()

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading index...")
index = faiss.read_index("vector_store/thesis.index")

with open("vector_store/thesis.index.meta.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    query_embedding = model.encode(
        [query.question],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        np.array(query_embedding), 5
    )

    retrieved_chunks = [metadata[i]["text"] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a thesis research assistant.
    Use ONLY the context provided.

    Context:
    {context}

    Question:
    {query.question}
    """

    # Call OpenAI or local LLM here
    return {
        "question": query.question,
        "context": context
    }

