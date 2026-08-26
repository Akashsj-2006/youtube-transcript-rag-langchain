# YouTube Transcript RAG

A RAG-based application that allows users to ask questions about
YouTube videos using their transcripts.

## How It Works

YouTube Video
↓
Transcript
↓
Text Splitting
↓
Ollama Embeddings
↓
FAISS Vector Database
↓
Similarity Search
↓
Relevant Transcript Chunks
↓
Ollama LLM
↓
Answer

## Technologies Used

- Python
- LangChain
- Ollama
- FAISS
- YouTube Transcript Loader
- Streamlit

## Setup

### 1. Clone the repository

git clone <your-repository-url>

### 2. Create a virtual environment

python -m venv .venv

### 3. Activate the virtual environment

Windows PowerShell:

.venv\Scripts\Activate.ps1

### 4. Install dependencies

pip install -r requirements.txt

### 5. Install Ollama models

ollama pull nomic-embed-text

## Run the Application

python main.py