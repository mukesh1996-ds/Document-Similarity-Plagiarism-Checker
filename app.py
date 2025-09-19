import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords


# -----------------------------
# Fetch webpage text
# -----------------------------
def fetch_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ")
        return text
    except Exception as e:
        return f"Error fetching URL: {e}"


# -----------------------------
# Text Cleaning & Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = text.strip()
    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens


# -----------------------------
# Compute Similarity (TF-IDF & BOW)
# -----------------------------
def compute_similarity_vectorizer(doc1, doc2, method="tfidf"):
    docs = [" ".join(doc1), " ".join(doc2)]

    if method == "tfidf":
        vectorizer = TfidfVectorizer()
    elif method == "bow":
        vectorizer = CountVectorizer()
    else:
        raise ValueError("Choose either 'tfidf' or 'bow'")

    vectors = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


# -----------------------------
# Compute Similarity (Word2Vec - Avg Embeddings)
# -----------------------------
def avg_word2vec(tokens, model):
    vecs = []
    for word in tokens:
        if word in model.key_to_index:
            vecs.append(model[word])
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)


def compute_similarity_word2vec(doc1, doc2, model):
    vec1 = avg_word2vec(doc1, model).reshape(1, -1)
    vec2 = avg_word2vec(doc2, model).reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("📖 Document Similarity & Plagiarism Checker")
    st.write("Compare two documents (via URLs) using **TF-IDF, BOW, and Word2Vec** similarity.")

    url1 = st.text_input("Enter first URL")
    url2 = st.text_input("Enter second URL")

    if st.button("Check Similarity"):
        if not url1 or not url2:
            st.warning("⚠️ Please enter both URLs.")
            return

        with st.spinner("Fetching and processing documents..."):
            text1 = fetch_text_from_url(url1)
            text2 = fetch_text_from_url(url2)

            if "Error" in text1 or "Error" in text2:
                st.error("❌ Error fetching one of the URLs.")
                return

            doc1 = clean_text(text1)
            doc2 = clean_text(text2)

            # Compute similarities
            tfidf_score = compute_similarity_vectorizer(doc1, doc2, method="tfidf")
            bow_score = compute_similarity_vectorizer(doc1, doc2, method="bow")

            st.info("⏳ Loading Word2Vec model (first time may take a while)...")
            model = api.load("glove-wiki-gigaword-100")

            word2vec_score = compute_similarity_word2vec(doc1, doc2, model)

        # Display results
        st.subheader("🔎 Similarity Scores")
        st.write(f"**TF-IDF Cosine Similarity:** {tfidf_score:.4f}")
        st.write(f"**BOW Cosine Similarity:** {bow_score:.4f}")
        st.write(f"**Word2Vec Cosine Similarity:** {word2vec_score:.4f}")

        # Plot comparison chart
        scores = [tfidf_score, bow_score, word2vec_score]
        labels = ["TF-IDF", "BOW", "Word2Vec"]

        fig, ax = plt.subplots()
        ax.bar(labels, scores, color=["#4CAF50", "#2196F3", "#FF9800"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similarity Score")
        ax.set_title("Document Similarity Comparison")

        st.pyplot(fig)


if __name__ == "__main__":
    main()