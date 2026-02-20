import os
import logging
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from ingestion.embedding import get_embedding_model  # <-- absolute import


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class IngestionPipeline:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        persist_path: str = "data/vector_store"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_path = persist_path
        self.embedding_model = get_embedding_model()

    def load_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"{pdf_path} not found.")

        logging.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    def split_documents(self, documents):
        logging.info("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)

    def build_vector_store(self, documents):
        logging.info("Building FAISS vector store...")
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        return vector_store

    def persist_vector_store(self, vector_store):
        os.makedirs(self.persist_path, exist_ok=True)
        logging.info(f"Saving vector store to {self.persist_path}")
        vector_store.save_local(self.persist_path)

    def run(self, pdf_path: str):
        documents = self.load_pdf(pdf_path)
        chunked_docs = self.split_documents(documents)
        vector_store = self.build_vector_store(chunked_docs)
        self.persist_vector_store(vector_store)

        logging.info("Ingestion completed successfully.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.ingestion_pipeline <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        pipeline = IngestionPipeline()
        pipeline.run(pdf_path)

    except Exception as e:
        logging.exception("Pipeline failed.")
        print(f"Error: {e}")
        sys.exit(1)