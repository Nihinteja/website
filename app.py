import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Title
st.title("🛒 Fake Product Review Detection")
st.write("Enter a product review to check whether it is REAL or FAKE")

# Load Dataset (sample dataset)
data = {
    "review": [
        "This product is amazing and works perfectly",
        "Worst product ever do not buy",
        "Absolutely fantastic highly recommend",
        "Fake product waste of money",
        "Very good quality and fast delivery",
        "Terrible item broke after one use",
        "Excellent purchase very satisfied",
        "Do not trust this seller fake reviews",
        "Loved it great value for money",
        "Bad quality not worth it"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df["review"] = df["review"].apply(clean_text)

# Features and Labels
X = df["review"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train Model
model = LogisticRegression()
model.fit(X, y)

# UI Input
user_input = st.text_area("Enter your review here:")

if st.button("Check Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ This review is REAL")
        else:
            st.error("❌ This review is FAKE")