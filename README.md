<<<<<<< HEAD
# Sentiment_Analysis
Developed an AI-powered RAG chatbot that reads articles and classifies sentiment as positive, negative, or neutral. Integrated NLP, machine learning, and explainability features to highlight sentiment-bearing words and provide contextual insights. Built with Python, LangChain, Hugging Face, and Streamlit for an interactive and deployable solution.
=======
# Sentiment Analysis RAG Chatbot

An AI-powered chatbot system designed to read newspapers, magazine articles, and online publications, classify content sentiment (Positive, Negative, Neutral), highlight sentiment-bearing lexicons, and provide explainable AI outputs backed by Retrieval-Augmented Generation (RAG).

![Project Architecture](PROJECT_DESIGN.md)

---

## 🌟 Key Features
- **Retrieval-Augmented Generation (RAG):** Natural language Q&A over ingested documents using FAISS and LangChain.
- **Explainable Sentiment Engine:** Lexicon word highlighting and score breakdowns using NLTK, spaCy, and TextBlob.
- **Dual Classification Pipeline:** TF-IDF + Scikit-Learn classifier alongside Transformer models (DistilBERT/RoBERTa).
- **FastAPI REST API:** Async endpoints for document ingestion, analysis, and RAG retrieval.
- **Interactive UI:** Streamlit web interface with real-time sentiment gauges and document viewer.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Launch FastAPI Backend
```bash
uvicorn backend.app:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`

### 3. Launch Streamlit UI
```bash
streamlit run frontend/app.py
```
- Dashboard: `http://localhost:8501`

---

## 📂 Project Structure
```
Sentiment Analysis_New/
├── PROJECT_DESIGN.md       # Full architecture & resume design blueprint
├── README.md               # Quickstart guide
├── requirements.txt        # Python package dependencies
├── data_ingestion/         # Document ingestion & spaCy/NLTK preprocessing
│   └── preprocessor.py
├── sentiment/              # Lexicon scoring & explainable highlighting
│   ├── lexicon_analyzer.py
│   └── classifier.py
├── rag/                    # FAISS vector store & LangChain retrieval
│   ├── vector_store.py
│   └── rag_chain.py
├── backend/                # FastAPI application REST endpoints
│   └── app.py
└── frontend/               # Streamlit interactive UI dashboard
    └── app.py
```
>>>>>>> 38c4081 (Initial commit: Sentiment Analysis RAG Chatbot project design and implementation)
