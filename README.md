# Langchain RAG Tutorial

This repository contains a Retrieval-Augmented Generation (RAG) implementation using LangChain. It demonstrates how to build a simple question-answering system over your own documents.

## Updates from Original Tutorial

This code has been updated from the original tutorial to work with newer versions of the libraries. Changes include:

1. Updated imports (langchain_chroma instead of langchain_community.vectorstores)
2. Updated embeddings (langchain_huggingface instead of OpenAI due to API limits)
3. Removed db.persist() call as it's no longer needed with newer Chroma versions
4. Added fallback mechanisms for when embeddings or LLM calls fail
5. Implemented a rule-based response generation instead of using OpenAI's API
6. Lowered relevance threshold from 0.7 to 0.4 to include more results

## Setup

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download NLTK resources:

```bash
python download_nltk_data.py
```

4. Add your documents to the `source_documents` directory
5. Run the ingestion script to process and store your documents:

```bash
python ingest_data.py
```

6. Query your documents:

```bash
python query_data.py "Your question here?"
```

## Requirements

- Python 3.8+
- sentence-transformers (for HuggingFaceEmbeddings)
- langchain and related packages
- nltk (with required data downloads)
- protobuf==3.20.0 (to fix compatibility issues)

## Example

```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

## Notes

This implementation uses a rule-based approach for generating responses rather than relying on external APIs like OpenAI. This makes it more accessible for users who don't have API access or want to avoid usage costs.
