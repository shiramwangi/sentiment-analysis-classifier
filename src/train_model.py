import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(data_path='data/cleaned_reviews.csv'):
    df = pd.read_csv(data_path)
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_tfidf, y_train)
    preds = clf.predict(X_test_tfidf)

    print("Evaluation Metrics")
    print("------------------")
    print("Accuracy :", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='binary'))
    print("Recall   :", recall_score(y_test, preds, average='binary'))
    print("F1-Score :", f1_score(y_test, preds, average='binary'))

    with open('vectorizer.pkl', 'wb') as vf:
        pickle.dump(vectorizer, vf)
    with open('model.pkl', 'wb') as mf:
        pickle.dump(clf, mf)
    print("Saved model and vectorizer.")

if __name__ == "__main__":
    train()