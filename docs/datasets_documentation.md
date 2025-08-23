# Dataset Documentation

## Origin & Purpose
The dataset comprises movie reviews labeled as positive or negative. It is intended to train a sentiment analysis model to distinguish emotional tone in textual feedback.

## Structure & Contents
| Column Name | Type   | Description                       |
|-------------|--------|-----------------------------------|
| text        | String | Original review text              |
| label       | String | Sentiment label: 'positive' or 'negative' |
| clean_text  | String | Preprocessed version of `text`    |

Data lives in `data/raw/movies_reviews.csv`. Cleaned data (tokenized, stopwords removed) is saved to `data/processed/`.

## Limitations & Bias
- **Sampling bias**: The dataset may over-represent certain genres or demographics.
- **Label noise**: Human sentiment is subjective—labels may sometimes be ambiguous.
- **Domain drift**: Model may underperform on text styles outside the training domain (e.g., social media slang).
- **Ethical note**: Sentiment decisions can influence user perception—misclassifications should be used cautiously.

*(Inspired by best practices for data cards and model transparency, emphasizing the importance of dataset documentation in ML workflows.)*:contentReference[oaicite:0]{index=0}

