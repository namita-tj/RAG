import os
import sys
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} pages.")
        return documents
    except Exception as e:
        logging.exception("Error loading PDF.")
        raise RuntimeError(f"Failed to load PDF: {str(e)}")


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    if not documents:
        raise ValueError("No documents provided for splitting.")

    try:
        logging.info("Applying RecursiveCharacterTextSplitter...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split_documents(documents)
        logging.info(f"Successfully created {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logging.exception("Error during text splitting.")
        raise RuntimeError(f"Text splitting failed: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python splitter.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        docs = load_pdf(pdf_path)
        chunks = split_documents(docs)

        print(f"\nTotal chunks created: {len(chunks)}")
        print("\nPreview of first chunk:\n")
        print(chunks[0].page_content[:500])

    except Exception as error:
        print(f"\nExecution failed: {error}")
        sys.exit(1)