# Production RAG with LangChain & Vector Databases

This repository contains code examples and implementations for building production-ready Retrieval-Augmented Generation (RAG) pipelines using LangChain, LangGraph, and vector databases like Chroma.

## 📂 Project Structure

```text
├── Chromadb/
│   ├── app.py                # Direct integration and usage of raw ChromaDB client
│   └── vector_stores.py      # LangChain's Chroma integration (similarity search, metadata filtering, retriever setup)
├── docs/
│   └── langchain_demo.pdf    # Sample PDF for document loading demonstrations
├── document_loaders.py       # Helper functions to load documents (e.g., PyPDFLoader)
├── main.py                   # Testing LLM configurations (OpenAI GPT and Anthropic Claude)
├── pyproject.toml            # Project configuration and dependency specifications
├── uv.lock                   # Lockfile for reproducible environment setup
└── .gitignore                # Git ignore rules (configured to ignore secrets & local environments)
```

---

## ⚙️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.12+ (configured for Python >= 3.14 in `pyproject.toml`) and `uv` or `pip` installed.

### 2. Environment Setup
Create a `.env` file in the root directory to store your API keys. Since this file contains sensitive credentials, it is ignored by Git.

```env
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Installation
This project uses `uv` for lightning-fast, reproducible dependency management. You can sync the environment using:

```bash
# Sync dependencies and create a virtual environment (.venv)
uv sync
```

Alternatively, install using standard `pip`:

```bash
pip install -r pyproject.toml
```

---

## 🚀 Usage & Features

### 1. Test LLM Connection (`main.py`)
Verifies that your OpenAI and Anthropic integrations are working correctly.
```bash
uv run main.py
```

### 2. Document Loaders (`document_loaders.py`)
Demonstrates how to load PDFs into LangChain document formats.
```bash
uv run document_loaders.py
```

### 3. Raw Chroma DB client (`Chromadb/app.py`)
Shows how to interact directly with ChromaDB (without LangChain wrappers): creating collections, upserting documents, and performing raw queries.
```bash
uv run Chromadb/app.py
```

### 4. LangChain Vector Store Integration (`Chromadb/vector_stores.py`)
Illustrates production patterns for utilizing Vector Stores, including:
* **Chroma Basics**: Creating vector stores from local documents.
* **Similarity Search with Scores**: Finding relevant documents and converting distance metrics into similarity scores.
* **Metadata Filtering**: Querying embeddings with conditional constraints on document metadata.
* **Retrievers**: Creating basic similarity retrievers and MMR (Maximal Marginal Relevance) retrievers to fetch diverse documents.
* **Persistence**: Persisting and reloading vector databases to/from local storage.

To run the configured entry points in the vector store script:
```bash
uv run Chromadb/vector_stores.py
```
