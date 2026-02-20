import torch
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.nn.functional import softmax
import faiss

# === CONFIG ===
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
THRESHOLD = 1.5  # Uncertainty threshold

# === LOAD MODELS ===
embedder = SentenceTransformer(EMBED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)

# === LOAD EMBEDDINGS + INDEX ===
with open("data/embeddings/chunk_texts.json", "r") as f:
    chunk_texts = json.load(f)
embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# === UNCERTAINTY ESTIMATOR ===
def compute_entropy(logits):
    probs = softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
    return entropy.item()


# === QA FUNCTION ===
def answer_question_with_context(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end]))
    return answer


# === MAIN PIPELINE ===
def adaptive_table_qa(query, top_k=3):
    # Step 1: Estimate uncertainty (no context)
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    entropy = compute_entropy(outputs.start_logits)

    if entropy > THRESHOLD:
        # Step 2: Retrieve relevant chunks
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, top_k)
        top_chunks = [chunk_texts[i] for i in I[0]]

        # Step 3: Combine and answer
        combined_context = " ".join(top_chunks)
        answer = answer_question_with_context(query, combined_context)
        return {"answer": answer, "retrieved": True, "entropy": entropy}
    else:
        # Skip retrieval, answer directly
        answer = answer_question_with_context(query, "")
        return {"answer": answer, "retrieved": False, "entropy": entropy}

