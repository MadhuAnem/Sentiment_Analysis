import sys
import os

# Ensure project root directory is on Python search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from data_ingestion.preprocessor import TextPreprocessor
from sentiment.lexicon_analyzer import LexiconSentimentAnalyzer
from sentiment.classifier import MLTransformerClassifier
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGSentimentPipeline

# Page setup
st.set_page_config(
    page_title="Sentiment Analysis RAG Chatbot",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .main-header { font-size: 2.3rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 2rem; }
    .sentiment-box { padding: 15px; border-radius: 8px; margin-bottom: 10px; font-weight: 600; }
    .pos-badge { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .neg-badge { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .neu-badge { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
</style>
""", unsafe_allow_html=True)

# Initialize state objects
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()
    st.session_state.lexicon_analyzer = LexiconSentimentAnalyzer()
    st.session_state.classifier = MLTransformerClassifier(use_transformer=False)
    st.session_state.vector_manager = VectorStoreManager()
    st.session_state.rag_pipeline = RAGSentimentPipeline(st.session_state.vector_manager)
    st.session_state.ingested_count = 0

st.markdown('<div class="main-header">📰 Sentiment Analysis RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ingest articles, analyze sentiment with explainable word highlights, and perform RAG Q&A.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📥 Article Ingestion", "🔍 Sentiment & Explainability", "💬 RAG Chatbot"])

# TAB 1: ARTICLE INGESTION
with tab1:
    st.subheader("Ingest Newspaper & Magazine Articles")
    st.write("Paste your raw article text below to populate the vector database.")
    
    article_input = st.text_area("Article Content", height=200, placeholder="Paste article text here...")
    if st.button("Ingest Article into Vector DB", type="primary"):
        if article_input.strip():
            cleaned = st.session_state.preprocessor.clean_text(article_input)
            st.session_state.vector_manager.add_documents([cleaned])
            st.session_state.ingested_count += 1
            st.success(f"Article successfully ingested and indexed! Total articles in DB: {st.session_state.ingested_count}")
        else:
            st.warning("Please enter article text before ingesting.")

# TAB 2: SENTIMENT & EXPLAINABILITY
with tab2:
    st.subheader("Explainable Sentiment Analysis")
    st.write("Analyze text polarity, model confidence, and view sentiment-bearing lexicon highlights.")
    
    sample_text = st.text_area("Input Text to Analyze", height=150, value="The company reported outstanding quarterly revenue growth and unprecedented profits, surpassing market expectations despite minor supply chain disruptions.")
    
    if st.button("Analyze Sentiment"):
        if sample_text.strip():
            analysis = st.session_state.lexicon_analyzer.analyze(sample_text)
            ml_res = st.session_state.classifier.classify(sample_text)
            highlight_html = st.session_state.lexicon_analyzer.generate_html_highlights(sample_text)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                label = analysis['label']
                badge_class = "pos-badge" if label == "Positive" else ("neg-badge" if label == "Negative" else "neu-badge")
                st.markdown(f'<div class="sentiment-box {badge_class}">Lexicon Label: {label}</div>', unsafe_allow_html=True)
            with col2:
                st.metric(label="Compound Polarity Score", value=f"{analysis['compound']:.4f}")
            with col3:
                st.metric(label="Classifier Label & Confidence", value=ml_res['label'], delta=f"{ml_res['confidence']*100:.1f}%")

            st.markdown("---")
            st.subheader("💡 Explainability: Highlighted Sentiment Lexicons")
            st.write("Green = Positive lexicon | Red = Negative lexicon")
            st.markdown(f'<div style="background-color: #F8FAFC; padding: 20px; border-radius: 8px; line-height: 1.8; font-size: 1.1rem;">{highlight_html}</div>', unsafe_allow_html=True)
            
            with st.expander("Detailed Keyword Extraction"):
                st.write("**Positive Keywords:**", analysis['positive_keywords'])
                st.write("**Negative Keywords:**", analysis['negative_keywords'])

# TAB 3: RAG CHATBOT
with tab3:
    st.subheader("Interactive RAG Q&A Chatbot")
    st.write("Ask questions about ingested articles. Contextual sentiment will be evaluated automatically.")
    
    query = st.text_input("Ask a question about your ingested documents:")
    if st.button("Submit Query"):
        if query.strip():
            res = st.session_state.rag_pipeline.query(query)
            st.markdown(f"**Aggregate Context Sentiment:** `{res['aggregate_sentiment']}`")
            st.info(res['answer'])
            
            if res['sources']:
                st.subheader("Retrieved Source Passages & Sentiment Breakdown")
                for idx, src in enumerate(res['sources']):
                    with st.expander(f"Source Passage #{idx+1} ({src['sentiment']} - Compound: {src['compound_score']:.4f})"):
                        st.markdown(src['highlighted_html'], unsafe_allow_html=True)
        else:
            st.warning("Please enter a query.")
