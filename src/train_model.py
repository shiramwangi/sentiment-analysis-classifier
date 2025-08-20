"""
train_model.py — Train sentiment classifier using TF-IDF and Naive Bayes.
Requires: pandas, scikit-learn
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def train(data_path='data/movie_reviews.csv'):
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_tfidf, y_train)
    preds = clf.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='weighted'))
    print("Recall:", recall_score(y_test, preds, average='weighted'))
    print("F1 Score:", f1_score(y_test, preds, average='weighted'))
    # Save model and vectorizer
    with open('model.pkl', 'wb') as mf:
        pickle.dump(clf, mf)
    with open('vectorizer.pkl', 'wb') as vf:
        pickle.dump(vectorizer, vf)

if __name__ == "__main__":
    train()