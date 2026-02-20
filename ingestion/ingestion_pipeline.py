import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class IngestionPipeline:
    """
    Handles document loading and text splitting.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: str) -> List:
        """
        Load PDF using LangChain PyPDFLoader.
        """
        if not os.path.exists(pdf_path):
            logging.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"{pdf_path} does not exist.")

        try:
            logging.info(f"Loading PDF from {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} pages successfully.")
            return documents

        except Exception as e:
            logging.exception("PDF loading failed.")
            raise RuntimeError(f"Error loading PDF: {str(e)}")

    def split_documents(self, documents: List) -> List:
        """
        Apply RecursiveCharacterTextSplitter.
        """
        if not documents:
            raise ValueError("No documents provided for splitting.")

        try:
            logging.info("Applying RecursiveCharacterTextSplitter...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            logging.info(f"Created {len(chunks)} chunks.")
            return chunks

        except Exception as e:
            logging.exception("Text splitting failed.")
            raise RuntimeError(f"Error splitting documents: {str(e)}")

    def run(self, pdf_path: str) -> List[Document]:
        """
        Complete ingestion pipeline.
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion_pipeline.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        pipeline = IngestionPipeline()
        chunks = pipeline.run(pdf_path)

        print(f"\nTotal chunks created: {len(chunks)}")
        print("\nPreview of first chunk:\n")
        print(chunks[0].page_content[:500])

    except Exception as error:
        print(f"\nExecution failed: {error}")
        sys.exit(1)