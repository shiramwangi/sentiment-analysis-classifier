# Sentiment Analysis Classifier

A modern, reproducible machine learning project for text sentiment analysis using Python, scikit-learn, and Flask.

## Project Overview

This repository provides a full workflow for building, training, evaluating, and deploying a sentiment analysis classifier. It covers text preprocessing, model training (TF-IDF + Naive Bayes), prediction, and a simple web UI for live testing.

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/shiramwangi/sentiment-analysis-classifier.git
   cd sentiment-analysis-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess your data**
   ```bash
   python src/preprocess.py
   ```

4. **Train the model**
   ```bash
   python src/train_model.py
   ```

5. **Make predictions**
   ```bash
   python src/predict.py "Your text here"
   ```

6. **Run the web app**
   ```bash
   python app/app.py
   ```

## Folder Structure

```
data/
  raw/        # Store original/raw datasets
  processed/  # Store cleaned/processed data
models/       # Saved models & vectorizers
notebooks/    # Jupyter notebooks (EDA, prototyping)
results/      # Metrics, plots, sample outputs
src/          # Source code (preprocessing, training, etc)
tests/        # Unit/integration tests
docs/         # Documentation, data dictionary, diagrams
app/          # Flask web app
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Mwangi Chiira (2025)
