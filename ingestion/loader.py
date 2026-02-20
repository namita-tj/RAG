import os
import sys
import logging
from langchain_community.document_loaders import PyPDFLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_pdf(pdf_path: str):
    """
    Loads a PDF document using LangChain's PyPDFLoader.
    Returns loaded documents.
    """

    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    try:
        logging.info(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        logging.info(f"Successfully loaded {len(documents)} pages.")
        return documents

    except Exception as e:
        logging.exception("Failed to load PDF.")
        raise RuntimeError(f"Error loading PDF: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python loader.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        docs = load_pdf(pdf_path)
        print(f"\nNumber of pages loaded: {len(docs)}")
        print("\nPreview of first page:\n")
        print(docs[0].page_content[:500])

    except Exception as error:
        print(f"\nExecution failed: {error}")
        sys.exit(1)