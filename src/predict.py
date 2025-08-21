import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

def predict(text: str, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    with open(vectorizer_path, 'rb') as vf, open(model_path, 'rb') as mf:
        vectorizer = pickle.load(vf)
        clf = pickle.load(mf)
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    print(f"Input: {text}")
    print(f"Predicted Sentiment: {pred}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your text here\"")
    else:
        predict(sys.argv[1])