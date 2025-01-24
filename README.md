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



Running the Application
Initialize Pinecone Client and Index:

The code initializes Pinecone with your API key, checks if the required index exists, and creates it if necessary.
The index is used to store the dense vector representations (embeddings) and sparse BM25 values.
Embedding Creation:

HuggingFace Embeddings: The all-MiniLM-L6-v2 model is used for creating dense vector embeddings of the text.
BM25 Encoding: The BM25 algorithm is used for sparse retrieval. It generates a set of BM25 values for the provided documents, which is stored and used for searching.
Hybrid Search Retriever:

The PineconeHybridSearchRetriever combines both dense embeddings (from HuggingFace) and sparse retrieval (from BM25) to fetch the most relevant documents based on a query.
Adding Texts and Querying:

The retriever adds some example texts (cities visited) to the Pinecone index.
A query like "What city did I visit first?" is used to demonstrate the hybrid search and retrieval process.
Code Overview
Pinecone Initialization:
The Pinecone client is initialized using the provided API key, and a vector index is created with the specified parameters (dimension size of embeddings and distance metric).

python
Copy
Edit
pc = Pinecone(api_key=api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimensionality of the embeddings
        metric="dotproduct",  # Sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
Embedding and Tokenization:
HuggingFace embeddings and NLTK tokenization are used to process the text:

python
Copy
Edit
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
BM25 Sparse Encoding:
BM25Encoder is used to encode documents into sparse representations and then stored in Pinecone:

python
Copy
Edit
from pinecone_text.sparse import BM25Encoder

bm25_encoder = BM25Encoder().default()
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")
Hybrid Search Retriever:
Combining dense and sparse search, the PineconeHybridSearchRetriever retrieves relevant documents:

python
Copy
Edit
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
retriever.add_texts(["In 2023, I visited Paris", "In 2022, I visited New York", "In 2021, I visited New Orleans"])
retriever.invoke("What city did I visit first?")

**Example Output**

 - After running the above code, the retriever will output the most relevant documents based on the query:

  Query: "What city did I visit first?"
  Output: The retriever will return the relevant document, in this case, "In 2023, I visited Paris."

**Conclusion**
  - This project demonstrates how hybrid search can be effectively implemented using Pinecone DB and Langchain. By combining dense embeddings and sparse 
  retrieval, it provides more accurate and contextually relevant search results. This approach is suitable for various applications, including document retrieval, 
  question answering, and information retrieval tasks.




