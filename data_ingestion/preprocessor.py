import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """Handles text cleaning, tokenization, stopword removal, and normalization."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Removes URLs, non-alphanumeric noise, and standardizes whitespace."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_remove_stopwords(self, text: str) -> list[str]:
        """Tokenizes cleaned text and removes stopwords."""
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned.lower())
        filtered_tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        return filtered_tokens

    def preprocess_document(self, text: str) -> dict:
        """Runs full preprocessing pipeline on input article text."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_remove_stopwords(cleaned)
        return {
            "raw_length": len(text),
            "cleaned_text": cleaned,
            "tokens": tokens,
            "word_count": len(tokens)
        }
