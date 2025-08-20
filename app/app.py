"""
Streamlit app for live sentiment predictions.
"""

import streamlit as st
import pickle

st.title("Sentiment Classifier")

user_text = st.text_input("Enter text to analyze:")

if st.button("Analyze"):
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    clf = pickle.load(open('model.pkl', 'rb'))
    X = vectorizer.transform([user_text])
    pred = clf.predict(X)[0]
    st.write("Predicted Sentiment:", pred)