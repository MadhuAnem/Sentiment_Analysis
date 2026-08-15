# 📰 Sentiment Analysis RAG Chatbot

> An enterprise-grade AI system that ingests unstructured news articles, magazine publications, and editorial text to perform **Retrieval-Augmented Generation (RAG)** Q&A alongside **Explainable Hybrid Sentiment Analysis** (Positive, Negative, Neutral).

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-0467DF?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## 🌟 Key Features

- 🧠 **Retrieval-Augmented Generation (RAG):** Context-aware Q&A over ingested document archives using **FAISS** vector indexing and **Sentence-Transformers** (`all-MiniLM-L6-v2`).
- 💡 **Explainable Sentiment Engine:** Solves LLM black-box issues by generating dynamic, color-coded HTML text highlights for positive (<mark style="background-color:#d4edda;color:#155724;">green</mark>) and negative (<mark style="background-color:#f8d7da;color:#721c24;">red</mark>) sentiment lexicons.
- ⚡ **Dual Sentiment Classifier:** Combines fast rule-based lexicons (**NLTK VADER**, **TextBlob**) with ML classifiers (**Scikit-learn TF-IDF + LogisticRegression**) and neural transformers (**DistilBERT SST-2**).
- 🔌 **Asynchronous REST API:** Production-ready **FastAPI** backend exposing `/ingest`, `/analyze`, and `/query` endpoints with interactive OpenAPI/Swagger docs.
- 🖥️ **Interactive Web Interface:** Feature-rich **Streamlit** dashboard with article uploaders, sentiment gauges, explainability viewers, and chat interfaces.

---

## 🏗️ System Architecture & Data Flow

```
                               ┌──────────────────────────┐
                               │  News & Magazine Text    │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │ Text Preprocessor        │
                               │ (clean, tokenize, stop)  │
                               └────────────┬─────────────┘
                                            │
                 ┌──────────────────────────┴──────────────────────────┐
                 │                                                     │
                 ▼                                                     ▼
┌─────────────────────────────────┐                 ┌──────────────────────────────────┐
│  Hybrid Sentiment &             │                 │  Dense Vector Embedding          │
│  Explainability Engine          │                 │  (Sentence-Transformers MiniLM)  │
│  • NLTK VADER & TextBlob        │                 └──────────────────┬───────────────┘
│  • Lexicon HTML Highlighter     │                                    │
│  • DistilBERT Transformer       │                                    ▼
└────────────────┬────────────────┘                 ┌──────────────────────────────────┐
                 │                                  │  FAISS Vector Store Index        │
                 │                                  └──────────────────┬───────────────┘
                 │                                                     │
                 └──────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │  RAG Retrieval Pipeline  │
                               │  (LangChain + Top-K=3)   │
                               └────────────┬─────────────┘
                                            │
                 ┌──────────────────────────┴──────────────────────────┐
                 │                                                     │
                 ▼                                                     ▼
┌─────────────────────────────────┐                 ┌──────────────────────────────────┐
│ ⚡ FastAPI REST Backend Server   │                 │ 🖥️ Streamlit Interactive UI      │
│ http://localhost:8000           │                 │ http://localhost:8501            │
└─────────────────────────────────┘                 └──────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
C:\Madhu\Sentiment Analysis_New\
├── .gitignore                  # Git ignore rules for virtualenvs, caches & IDE files
├── PROJECT_DESIGN.md           # Architectural blueprint & resume guide
├── PROJECT_DOCUMENTATION.md    # Master technical manual & deep-dive specification
├── README.md                   # Project overview & quickstart manual (this file)
├── requirements.txt            # Python dependencies manifest
│
├── data_ingestion/             # Data cleaning & preprocessing layer
│   ├── __init__.py
│   └── preprocessor.py         # Regex cleaning, URL stripping, tokenization & stopwords
│
├── sentiment/                  # Sentiment analysis & explainability layer
│   ├── __init__.py
│   ├── lexicon_analyzer.py     # NLTK VADER, TextBlob scoring & HTML word highlighter
│   └── classifier.py           # TF-IDF + LogisticRegression / DistilBERT classification
│
├── rag/                        # Vector DB indexing & LangChain retrieval layer
│   ├── __init__.py
│   ├── vector_store.py         # FAISS vector database manager & MiniLM embeddings
│   └── rag_chain.py            # LangChain retrieval pipeline & Q&A context synthesizer
│
├── backend/                    # Production REST API microservice
│   ├── __init__.py
│   └── app.py                  # FastAPI server with /ingest, /analyze, and /query endpoints
│
└── frontend/                   # Web user interface dashboard
    ├── __init__.py
    └── app.py                  # Streamlit dual-tab dashboard & visual explainability viewer
```

---

## 🧰 Tech Stack & Tools Used

| Domain | Technologies & Libraries | Purpose |
| :--- | :--- | :--- |
| **Language** | **Python 3.10+** | Core programming language. |
| **NLP & Text** | **NLTK, spaCy, TextBlob** | Lexicon scoring, tokenization, lemmatization & stopword removal. |
| **Machine Learning** | **Scikit-learn, PyTorch** | TF-IDF vectorization, Logistic Regression & tensor calculations. |
| **Transformers** | **Hugging Face Transformers** | DistilBERT SST-2 neural sentiment model. |
| **Vector DB** | **FAISS (`faiss-cpu`)** | Dense vector similarity indexing & sub-millisecond search. |
| **RAG Pipeline** | **LangChain, Sentence-Transformers** | Passage chunking (`all-MiniLM-L6-v2`) and retrieval Q&A chains. |
| **Backend API** | **FastAPI, Uvicorn, Pydantic** | Asynchronous REST service exposing high-performance endpoints. |
| **Frontend UI** | **Streamlit** | Interactive web dashboard with HTML styling and tab navigation. |

---

## 🚀 Step-by-Step Operating & Setup Guide

### 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed on your system.

### 2. Virtual Environment Setup
Clone the repository and set up a clean Python virtual environment:

```bash
# Navigate into the project folder
cd "C:\Sentiment Analysis_New"

# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt` and download spaCy language models:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Running the Backend REST API Server (FastAPI)
Launch the FastAPI backend server using Uvicorn:

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
- **REST API Base URL:** `http://localhost:8000`
- **Interactive Swagger Documentation:** [`http://localhost:8000/docs`](http://localhost:8000/docs)

### 5. Running the Frontend Interactive Web App (Streamlit)
Open a new terminal window, activate your virtual environment, and launch Streamlit:

```bash
streamlit run frontend/app.py
```
- **Web UI Dashboard:** [`http://localhost:8501`](http://localhost:8501)

---

## 📡 REST API Reference

The FastAPI backend exposes 3 key REST endpoints:

### `POST /ingest`
Preprocesses and indexes input articles into the FAISS vector database.
```json
// Request Payload:
{
  "articles": [
    "The quarterly earnings report showed unprecedented revenue growth, surging 25% year-over-year despite market headwinds."
  ]
}

// Response Payload:
{
  "status": "success",
  "ingested_count": 1,
  "message": "Successfully ingested 1 articles into vector store."
}
```

### `POST /analyze`
Runs dual lexicon + ML sentiment analysis and returns HTML explainability highlights.
```json
// Request Payload:
{
  "text": "The company reported record profits but faced minor supply chain delays."
}

// Response Payload:
{
  "status": "success",
  "lexicon_analysis": {
    "label": "Positive",
    "compound": 0.4588,
    "positive_keywords": ["record", "profits"],
    "negative_keywords": ["delays"]
  },
  "ml_classification": {
    "label": "Positive",
    "confidence": 0.9231,
    "model_type": "ML (TF-IDF + LogisticRegression)"
  },
  "explainability_html": "The company reported <mark style=\"background-color: #d4edda;\">record</mark> <mark style=\"background-color: #d4edda;\">profits</mark> but faced minor supply chain <mark style=\"background-color: #f8d7da;\">delays</mark>."
}
```

### `POST /query`
Performs top-K semantic similarity search over FAISS indices and summarizes contextual sentiment.
```json
// Request Payload:
{
  "query": "What were the financial results?"
}

// Response Payload:
{
  "status": "success",
  "result": {
    "query": "What were the financial results?",
    "aggregate_sentiment": "Positive",
    "answer": "Based on the ingested articles...",
    "sources": [...]
  }
}
```

---

## 📄 Project Documentation Links

- 📘 **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md):** Complete technical manual, architectural design, data contracts, and cloud deployment guide.
- 📐 **[PROJECT_DESIGN.md](PROJECT_DESIGN.md):** Architecture design document & step-by-step implementation blueprint.
