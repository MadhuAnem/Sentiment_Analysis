import sys
import os

# Ensure project root directory is on Python search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_ingestion.preprocessor import TextPreprocessor
from sentiment.lexicon_analyzer import LexiconSentimentAnalyzer
from sentiment.classifier import MLTransformerClassifier
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGSentimentPipeline

app = FastAPI(
    title="Sentiment Analysis RAG Chatbot API",
    description="REST API for article ingestion, sentiment classification, explainable term highlighting, and RAG Q&A.",
    version="1.0.0"
)

# Initialize core microservices
preprocessor = TextPreprocessor()
lexicon_analyzer = LexiconSentimentAnalyzer()
classifier = MLTransformerClassifier(use_transformer=False)
vector_store_manager = VectorStoreManager()
rag_pipeline = RAGSentimentPipeline(vector_store_manager)


class IngestRequest(BaseModel):
    articles: list[str]


class AnalyzeRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {
        "system": "Sentiment Analysis RAG Chatbot API",
        "status": "active",
        "endpoints": ["/ingest", "/analyze", "/query"]
    }


@app.post("/ingest")
def ingest_documents(req: IngestRequest):
    """Preprocesses and indexes input articles into FAISS vector database."""
    if not req.articles:
        raise HTTPException(status_code=400, detail="Article list cannot be empty.")
    
    cleaned_articles = [preprocessor.clean_text(a) for a in req.articles]
    vector_store_manager.add_documents(cleaned_articles)
    return {
        "status": "success",
        "ingested_count": len(cleaned_articles),
        "message": f"Successfully ingested {len(cleaned_articles)} articles into vector store."
    }


@app.post("/analyze")
def analyze_sentiment(req: AnalyzeRequest):
    """Runs dual lexicon + ML classification and returns explainability highlighted HTML."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
    
    preprocessed = preprocessor.preprocess_document(req.text)
    lexicon_res = lexicon_analyzer.analyze(req.text)
    html_highlight = lexicon_analyzer.generate_html_highlights(req.text)
    ml_res = classifier.classify(req.text)

    return {
        "status": "success",
        "preprocessing": preprocessed,
        "lexicon_analysis": lexicon_res,
        "ml_classification": ml_res,
        "explainability_html": html_highlight
    }


@app.post("/query")
def query_rag_chatbot(req: QueryRequest):
    """Executes RAG search over ingested articles and returns sentiment-annotated answers."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = rag_pipeline.query(req.query)
    return {
        "status": "success",
        "result": result
    }
