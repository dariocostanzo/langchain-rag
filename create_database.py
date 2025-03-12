# Update imports section
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_chroma import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    try:
        # Try using HuggingFaceEmbeddings with a simpler model
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")
        # Fallback to OpenAI embeddings if API key is available
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            embedding_function = OpenAIEmbeddings(
                openai_api_key=api_key,
                base_url="https://api.openai.com/v1"
            )
        else:
            # Last resort fallback
            from langchain_community.embeddings import FakeEmbeddings
            print("Falling back to FakeEmbeddings for testing purposes")
            embedding_function = FakeEmbeddings(size=384)
    
    # Create a new DB from the documents - remove the persist() call
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    # Remove the db.persist() line as it's no longer needed/supported
    # The persist_directory parameter in the constructor handles persistence
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
