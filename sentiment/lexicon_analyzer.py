import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class LexiconSentimentAnalyzer:
    """Combines NLTK VADER and TextBlob for lexicon analysis and explainability word highlighting."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        """Computes sentiment scores, extracts polarity, and identifies key sentiment terms."""
        vader_scores = self.vader.polarity_scores(text)
        blob = TextBlob(text)
        
        words = text.split()
        positive_words = []
        negative_words = []
        
        for w in words:
            clean_w = w.strip(".,!?\"'()[]")
            if not clean_w:
                continue
            pol = TextBlob(clean_w).sentiment.polarity
            if pol > 0.2:
                positive_words.append(clean_w)
            elif pol < -0.2:
                negative_words.append(clean_w)

        compound = vader_scores['compound']
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "label": label,
            "compound": compound,
            "vader_breakdown": vader_scores,
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "positive_keywords": list(set(positive_words)),
            "negative_keywords": list(set(negative_words))
        }

    def generate_html_highlights(self, text: str) -> str:
        """Generates HTML snippet with highlighted positive (green) and negative (red) words for explainability."""
        words = text.split()
        highlighted = []
        for word in words:
            clean_word = word.strip(".,!?\"'()[]")
            pol = TextBlob(clean_word).sentiment.polarity if clean_word else 0
            if pol > 0.2:
                highlighted.append(f'<mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 3px;">{word}</mark>')
            elif pol < -0.2:
                highlighted.append(f'<mark style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 3px;">{word}</mark>')
            else:
                highlighted.append(word)
        return " ".join(highlighted)
