"""
preprocess.py — text preprocessing for sentiment analysis:
- Lowercasing
- Tokenization
- Stopword removal

Libraries: NLTK, pandas
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Example setup
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
    return " ".join(filtered)

# Example usage:
# df['clean_text'] = df['text'].apply(clean_text)