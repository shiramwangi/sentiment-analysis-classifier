import pandas as pd
from src.preprocess import preprocess_file
from src.train_model import train
from src.predict import predict

def test_pipeline(tmp_path, monkeypatch):
    # Create a tiny DataFrame
    df = pd.DataFrame({
        "text": ["I love this movie", "I hate this movie"],
        "label": ["positive", "negative"]
    })
    in_file = tmp_path / "movies_reviews.csv"
    df.to_csv(in_file, index=False)
    preprocess_file(input_csv=str(in_file), output_csv=str(tmp_path / "cleaned.csv"))
    # Training with cleaned data
    monkeypatch.chdir(tmp_path)
    train(data_path=str(tmp_path / "cleaned.csv"))
    # Predict
    predict("I love this movie", model_path="model.pkl", vectorizer_path="vectorizer.pkl")

