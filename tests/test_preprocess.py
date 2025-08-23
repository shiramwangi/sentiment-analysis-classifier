from src.preprocess import clean_text

def test_clean_text_basic():
    assert clean_text("Hello, WORLD!") == "hello world"

def test_clean_text_stopwords():
    text = "Mwangi Tests classifier"
    cleaned = clean_text(text)
    assert "this" not in cleaned and "is" not in cleaned

