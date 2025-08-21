import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
    return " ".join(filtered)

def preprocess_file(input_csv='data/movie_reviews.csv', output_csv='data/cleaned_reviews.csv'):
    df = pd.read_csv(input_csv)
    df['clean_text'] = df['text'].apply(clean_text)
    df[['text', 'clean_text', 'label']].to_csv(output_csv, index=False)
    print(f"Saved cleaned data to {output_csv}. Sample:")
    print(df.head())

if __name__ == "__main__":
    preprocess_file()