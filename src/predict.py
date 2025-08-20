"""
predict.py — Predict sentiment for a given text using trained model.
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def predict(text: str, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    with open(vectorizer_path, 'rb') as vf:
        vectorizer = pickle.load(vf)
    with open(model_path, 'rb') as mf:
        clf = pickle.load(mf)
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    print(f"Sentiment: {pred}")

if __name__ == "__main__":
    import sys
    predict(sys.argv[1])