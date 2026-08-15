from rag.vector_store import VectorStoreManager
from sentiment.lexicon_analyzer import LexiconSentimentAnalyzer

class RAGSentimentPipeline:
    """Combines vector retrieval with sentiment extraction for contextual Q&A."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.lexicon_analyzer = LexiconSentimentAnalyzer()

    def query(self, user_query: str) -> dict:
        """Retrieves top passages, analyzes their sentiment, and synthesizes answer context."""
        retrieved_docs = self.vector_store_manager.similarity_search(user_query, top_k=3)
        
        if not retrieved_docs:
            return {
                "query": user_query,
                "answer": "No documents ingested in vector store yet. Please ingest articles first.",
                "sources": [],
                "aggregate_sentiment": "Neutral"
            }

        sources = []
        sentiments = []
        for doc in retrieved_docs:
            analysis = self.lexicon_analyzer.analyze(doc.page_content)
            highlighted = self.lexicon_analyzer.generate_html_highlights(doc.page_content)
            sentiments.append(analysis['label'])
            sources.append({
                "content": doc.page_content,
                "highlighted_html": highlighted,
                "sentiment": analysis['label'],
                "compound_score": analysis['compound'],
                "positive_words": analysis['positive_keywords'],
                "negative_words": analysis['negative_keywords']
            })

        # Calculate overall sentiment frequency
        pos_count = sentiments.count("Positive")
        neg_count = sentiments.count("Negative")
        agg_sentiment = "Positive" if pos_count > neg_count else ("Negative" if neg_count > pos_count else "Neutral")

        # Synthesize informative response text
        context_summary = "\n\n".join([f"[Source {i+1} - {s['sentiment']}]: {s['content']}" for i, s in enumerate(sources)])
        answer = f"Based on the ingested articles, here is the relevant context:\n\n{context_summary}"

        return {
            "query": user_query,
            "answer": answer,
            "aggregate_sentiment": agg_sentiment,
            "sources": sources
        }
