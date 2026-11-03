import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STEMMER = PorterStemmer()
_STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    """Lowercase, remove punctuation/numbers, remove stopwords and stem."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [
        _STEMMER.stem(word)
        for word in text.split()
        if word not in _STOP_WORDS and len(word) > 1
    ]
    return " ".join(words)
