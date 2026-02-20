import sys
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loader import load_pdf  # Import from loader.py


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def perform_chunking(documents: List, chunk_size=800, chunk_overlap=150):
    """
    Applies Recursive Character-Based Chunking.
    Designed for RAG pipelines.
    """

    if not documents:
        raise ValueError("No documents provided for chunking.")

    try:
        logging.info("Applying RecursiveCharacterTextSplitter...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        logging.info(f"Created {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logging.exception("Chunking failed.")
        raise RuntimeError(f"Chunking error: {str(e)}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python splitter.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        docs = load_pdf(pdf_path)  # Using loader.py
        chunks = perform_chunking(docs)

        print(f"\nTotal chunks created: {len(chunks)}")
        print("\nPreview of first chunk:\n")
        print(chunks[0].page_content[:500])

    except Exception as error:
        print(f"\nExecution failed: {error}")
        sys.exit(1)