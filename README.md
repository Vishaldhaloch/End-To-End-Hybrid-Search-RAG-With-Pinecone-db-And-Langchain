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


## Setup Environment
**Get API Keys:**
**Pinecone:**
  - Create an account at Pinecone, then retrieve your API Key.
**HuggingFace:**
 - Create an account at HuggingFace and obtain a token for using pre-trained embeddings.

Set Up Environment Variables:
Create a .env file in the project directory and add your API keys:
HF_TOKEN=your_huggingface_api_key


   
2. **Create a Conda Environment:**
    ```bash
  conda create -n summarize-content python=3.8 -y
   
3. **Activate the Environment:**
    ```bash
    conda activate summarize-content
  
4. **Install Dependencies:**
   Install the required libraries using requirements.txt:
    ```bash 
    pip install -r requirements.txt
   
5. **Set Up Environment Variables:**
   Create a .env file and add your Groq API Key:
    ```bash
    GROQ_API_KEY=your_groq_api_key

  
1. **How to Run**
   Start the task execution by running the following Python script:
     ```bash
     streamlit run app.py
   
2. **Use the Application:**
    - Open the Streamlit app in your browser.
    -  In the sidebar, enter your Groq API Key.
    -  Paste the URL of the YouTube video or website you want to summarize.
    -   Click the "Summarize the Content from YT or Website" button to generate a summary of the content.

3. **View the Results:**
    - The application will display the summarized text based on the content of the URL provided.
   
   **Code Overview**
1. **Groq API Integration:**
   - The ChatGroq class is used to interface with the Groq model (Gemma-7b-It), which is responsible for generating summaries of the content.

2. **Content Loading:**
    . YouTube:
       - If the URL is a YouTube video, the YoutubeLoader extracts the video transcript and metadata.
       
    . Website:
       - If the URL is a website, the UnstructuredURLLoader extracts the text content from the webpage.
3. **Prompt Template:**
     - The prompt template instructs the model to provide a summary of the content in 300 words. The extracted text is fed into this prompt to generate the summary.
4. **Summarization Chain**
    - The LangChain summarization chain takes the loaded content and uses the Groq LLM model to summarize it based on the prompt.
