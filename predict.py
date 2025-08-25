import sys
import pickle
import os

def load_model():
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Trained model/vectorizer not found. Please run train_model.py first.")

    with open(model_path, "rb") as mf:
        model = pickle.load(mf)
    with open(vectorizer_path, "rb") as vf:
        vectorizer = pickle.load(vf)
    
    return model, vectorizer

def predict_sentiment(text):
    model, vectorizer = load_model()
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    return prediction, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<your review text>\"")
        sys.exit(1)

    review_text = sys.argv[1]
    prediction, confidence = predict_sentiment(review_text)
    print(f"Predicted sentiment: {prediction} ({confidence:.2f} confidence)")
