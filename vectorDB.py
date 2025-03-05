# Introduced a new module to integrate Chroma with
# Azure OpenAI embeddings for vector storage and retrieval.
# This includes functionality for loading TXT and PDF documents,
# storing them in a Chroma database, and retrieving them
# with relevant queries.

from config import azure_settings
import os
from llms.gptllm import logger  # or define a new logger
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

COLLECTION_NAME = 'collection'
DOC_PATH = 'documents'
DB_DIRECTORY = 'chroma_db'

# Create embeddings using Azure OpenAI
embedding_function = AzureOpenAIEmbeddings(
    model=azure_settings.azure_deployment_embedding
)
logger.info("Azure Embedding initialized.")

# Initialize ChromaDB
chroma_vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
    persist_directory=DB_DIRECTORY
)
logger.info("Chroma DB initialized.")


def check_documents(directory_path: str):
    """
    Load documents from a specified directory, supporting both TXT and PDF files.

    Args:
        directory_path (str): Path to the directory containing documents.

    Returns:
        List[Document]: List of loaded documents.
    """
    documents = []

    # Load TXT files
    txt_loader = DirectoryLoader(directory_path, glob="*.txt")
    try:
        txt_documents = txt_loader.load()
        documents.extend(txt_documents)
        logger.info(f"Loaded {len(txt_documents)} TXT documents.")
    except Exception as e:
        logger.error(f"Error loading TXT documents: {e}")

    # Load PDF files
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    for file_name in pdf_files:
        try:
            file_path = os.path.join(directory_path, file_name)
            pdf_documents = PyPDFLoader(file_path).load_and_split()
            documents.extend(pdf_documents)
            logger.info(f"Loaded {len(pdf_documents)} pages from {file_name}.")
        except Exception as e:
            logger.error(f"Error loading PDF file {file_name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def get_retriever(vector_store=chroma_vector_store):
    try:
        return vector_store.as_retriever()
    except Exception as e:
        logger.error(f"Error getting Chroma store as retriever: {e}")
        raise


def load_chroma_store(vector_store=chroma_vector_store):
    try:
        docs = check_documents(DOC_PATH)
        vector_store.add_documents(documents=docs)
        # vector_store.persist()
        logger.info("Chroma store successfully created and persisted.")
        print("Chroma store successfully created and data stored!")
    except Exception as e:
        logger.exception("Error creating or persisting Chroma store:", exc_info=e)


# Standalone use example of this script:
# 1. load some documents of the DOC_PATH
# 2. run this file for testing
if __name__ == "__main__":
    load_chroma_store()
    retriever = get_retriever()
    print(retriever.get_relevant_documents("What is the meaning of life?"))
