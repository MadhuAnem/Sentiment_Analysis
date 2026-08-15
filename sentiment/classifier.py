from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

class MLTransformerClassifier:
    """Hybrid Classifier using TF-IDF + LogisticRegression baseline or Transformer pipeline."""

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        if self.use_transformer:
            try:
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except Exception:
                self.use_transformer = False
        
        if not self.use_transformer:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.model = LogisticRegression()
            # Fit initial dummy baseline
            X_dummy = ["Great article, positive news", "Terrible crash, bad outcome", "The event occurred on Monday"]
            y_dummy = ["Positive", "Negative", "Neutral"]
            X_vec = self.vectorizer.fit_transform(X_dummy)
            self.model.fit(X_vec, y_dummy)

    def classify(self, text: str) -> dict:
        """Classifies text sentiment using Transformer or TF-IDF model."""
        if self.use_transformer:
            res = self.transformer_pipeline(text[:512])[0]
            label_map = {"POSITIVE": "Positive", "NEGATIVE": "Negative"}
            return {
                "label": label_map.get(res['label'], "Neutral"),
                "confidence": round(res['score'], 4),
                "model_type": "Transformer (DistilBERT)"
            }
        else:
            X_vec = self.vectorizer.transform([text])
            probs = self.model.predict_proba(X_vec)[0]
            pred = self.model.predict(X_vec)[0]
            confidence = float(max(probs))
            return {
                "label": pred,
                "confidence": round(confidence, 4),
                "model_type": "ML (TF-IDF + LogisticRegression)"
            }
