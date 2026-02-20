import os
import json
import argparse
from docx import Document

# ==============================
# CONFIG / DEFINITIONS
# ==============================
CHUNK_SIZE = 500       # words per chunk
OVERLAP = 100          # words overlap
DATA_DIR = "data"      # folder to store outputs
SUPPORTED_EXTENSIONS = [".txt", ".docx"]
DEFAULT_DOC_TYPE = "general"
DEFAULT_OUTPUT_FILE = "chunks.json"


# ==============================
# HELPER FUNCTIONS
# ==============================
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 1

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append((chunk_id, chunk_text))
        chunk_id += 1

        start += chunk_size - overlap

    return chunks


def process_file(file_path, doc_type=DEFAULT_DOC_TYPE):
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Only {SUPPORTED_EXTENSIONS} are supported")

    if ext == ".txt":
        text = read_txt(file_path)
    else:
        text = read_docx(file_path)

    chunks = split_into_chunks(text)

    structured_chunks = []
    for chunk_id, chunk_text in chunks:
        structured_chunks.append({
            "chunk_id": chunk_id,
            "source": os.path.basename(file_path),
            "document_type": doc_type,
            "text": chunk_text
        })

    return structured_chunks


# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Chunk documents into semantic chunks")
    parser.add_argument("input_file", help="Path to input .txt or .docx file")
    parser.add_argument("--doc_type", default=DEFAULT_DOC_TYPE, help="Type of document (literature, draft, paper, etc.)")
    parser.add_argument("--output", default=os.path.join(DATA_DIR, DEFAULT_OUTPUT_FILE), help="Output JSON file path")

    args = parser.parse_args()

    # Ensure output folder exists
    ensure_data_dir()

    # Process file
    chunks = process_file(args.input_file, args.doc_type)

    # Save chunks
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"Created {len(chunks)} semantic chunks from {args.input_file}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
