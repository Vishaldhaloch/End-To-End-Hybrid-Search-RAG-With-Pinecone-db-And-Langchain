# Hybrib-Search
# End-to-End Hybrid Search with Pinecone DB and Langchain

This project demonstrates how to perform hybrid search using **Pinecone DB** and **Langchain** to create a Retrieval-Augmented Generation (RAG) model. The application combines dense and sparse search techniques to find the most relevant documents and generate high-quality answers from them.

The integration of **Pinecone** for vector storage, **BM25** for sparse retrieval, and **HuggingFace Embeddings** for dense retrieval enables a robust hybrid search mechanism.

---

## Key Features

- **Hybrid Search**: Combines dense embeddings (via HuggingFace) and sparse retrieval (via BM25) for improved search quality.
- **Pinecone DB**: Uses **Pinecone** for efficient vector storage and retrieval.
- **Langchain**: Leverages **Langchain** for chaining together the search and generation pipelines.
- **End-to-End Search and Answer Generation**: From document storage to question answering using the hybrid search mechanism.

---

## Prerequisites

Before running the project, ensure you have the following:

1. **Python 3.7+**: Ensure you're running Python version 3.7 or above.
2. **Pinecone API Key**: You’ll need a **Pinecone** account and an **API Key**.
3. **HuggingFace API Key**: You need a **HuggingFace** account to use pre-trained embedding models.
4. **Langchain**: A framework to manage the language models, embedding models, and document retrieval.
5. **NLTK**: Used for tokenizing text in this example.

---

## Installation

To install the required libraries, run:

```bash
pip install --upgrade --quiet pinecone-client pinecone-text pinecone-notebooks
pip install langchain huggingface_hub langchain-huggingface nltk

**Setup Environment**
1.**Get API Keys:**

**Pinecone:**  - Create an account at Pinecone, then retrieve your API Key.
**HuggingFace:** -  Create an account at HuggingFace and obtain a token for using pre-trained embeddings.

2.**Set Up Environment Variables:**

Create a .env file in the project directory and add your API keys:
