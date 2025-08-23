import pandas as pd
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

def train(data_path='data/processed/movies_reviews.csv'):
    df = pd.read_csv(data_path)
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_tfidf, y_train)
    preds = clf.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label="positive"),
        "recall": recall_score(y_test, preds, pos_label="positive"),
        "f1_score": f1_score(y_test, preds, pos_label="positive"),
        "classification_report": classification_report(y_test, preds, output_dict=True)
    }

    # Save metrics to results folder
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if key != "classification_report":
            print(f"{key.capitalize()}: {value:.4f}")

    with open("vectorizer.pkl", "wb") as vf:
        pickle.dump(vectorizer, vf)
    with open("model.pkl", "wb") as mf:
        pickle.dump(clf, mf)

    print("Model and vectorizer saved, metrics stored in results/metrics.json")
