import pandas as pd
from pathlib import Path
import pickle
from src.preprocess import preprocess_file
from src.train_model import train
from src.predict import predict

def test_full_pipeline(tmp_path, capsys):
    # 1. Create sample dataset
    df = pd.DataFrame({
        "text": ["I love this movie", "I hate this movie"],
        "label": ["positive", "negative"]
    })
    in_csv = tmp_path / "movies_reviews.csv"
    df.to_csv(in_csv, index=False)

    # 2. Run preprocessing
    cleaned_csv = tmp_path / "cleaned.csv"
    preprocess_file(input_csv=str(in_csv), output_csv=str(cleaned_csv))

    # 3. Train using cleaned data
    model_csv = cleaned_csv  # using same path
    Path("vectorizer.pkl").unlink(missing_ok=True)
    Path("model.pkl").unlink(missing_ok=True)
    train(data_path=str(model_csv))

    # 4. Predict and capture output
    predict("I love this movie", model_path="model.pkl", vectorizer_path="vectorizer.pkl")
    captured = capsys.readouterr()
    assert "Sentiment: positive" in captured.out
