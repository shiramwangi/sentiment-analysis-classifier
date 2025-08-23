# Model Documentation

## Overview
This project uses a **TF-IDF vectorizer** combined with a **Multinomial Naive Bayes** classifier to predict sentiment from movie reviews.

## Feature Engineering
- **TF-IDF** emphasizes common yet meaningful words, reducing the impact of frequent but uninformative terms.
- Uses term frequency weighting to reflect each word’s importance within and across documents.

## Model Choice & Hyperparameters
- **Model**: Multinomial Naive Bayes — efficient and effective for high-dimensional, sparse text features.
- **Hyperparameters**:
  - `alpha = 1.0`: Laplace smoothing to handle unseen words.
  - `max_df = 0.7`: Ignores very common words.
- **Train-test split**: 80/20 with `random_state=42` for reproducibility.

## Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, **F1-Score** for the "positive" label.
- Full **classification report** is saved in `results/metrics.json`.

## Rationale
Multinomial Naive Bayes is a strong baseline for text tasks—fast, interpretable, and robust. TF-IDF complements it by weighting informative words effectively.

---

##  Limitations & Future Improvements
- Doesn't capture context or negation (e.g., “not good”).
- Could be enhanced with n-grams or deep learning models (e.g., RNNs, transformer-based).
- Feature importance or explainability tools (e.g., SHAP) can improve transparency.:contentReference[oaicite:1]{index=1}

