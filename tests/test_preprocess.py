from src.preprocess import clean_text

def test_clean_text_basic():
    assert clean_text("Hello, WORLD!") == "hello world"

def test_clean_text_stopwords_removed():
    # Words like "this" and "is" are common stop words and should be removed
    output = clean_text("This is a test of preprocessing.")
    assert "this" not in output and "is" not in output
