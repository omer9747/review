import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        cleaned_words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
        cleaned_sentences.append(' '.join(cleaned_words))
    
    return sentences, cleaned_sentences

def rank_sentences(sentences, cleaned_sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    cosine_matrix = cosine_similarity(tfidf_matrix)
    scores = np.mean(cosine_matrix, axis=1)
    
    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
    
    return ranked_sentences

def summarize_text(text, num_sentences=3):
    sentences, cleaned_sentences = preprocess_text(text)
    ranked_sentences = rank_sentences(sentences, cleaned_sentences)
    
    selected_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[2])
    summary = ' '.join([s[1] for s in selected_sentences])
    
    return summary

def main():
    st.title("📝 Text Summarizer")

    text = st.text_area("Enter the text to summarize:", height=300)
    num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)

    if st.button("Summarize"):
        if text:
            # Check if the input text is not numeric
            if any(char.isdigit() for char in text):
                st.error("🚨 Please enter valid text. Numbers are not allowed.")
            else:
                summary = summarize_text(text, num_sentences)
                st.subheader("Summary")
                st.write(summary)
        else:
            st.error("🚨 Please enter some text to summarize.")

if __name__ == "__main__":
    main()