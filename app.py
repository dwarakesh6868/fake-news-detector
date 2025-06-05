import streamlit as st
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Text Preprocessing Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- Load Vectorizer and Model ---
@st.cache_resource
def load_model():
    # Load vectorizer and model from saved .pkl files
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return vectorizer, model

vectorizer, model = load_model()

# --- Streamlit Web UI ---
st.title("📰 Fake News Detection App")
st.markdown("Enter news content below to check if it's **Fake** or **Real**.")

input_text = st.text_area("Paste the news article content here:")

if st.button("Detect"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        vector_input = vectorizer.transform([cleaned])
        prediction = model.predict(vector_input)[0]

        if prediction == 0:
            st.error("🚨 This is likely **FAKE NEWS**.")
        else:
            st.success("✅ This appears to be **REAL NEWS**.")
