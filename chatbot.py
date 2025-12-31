import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import faqs

questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

st.title("FAQ Chatbot")

user_input = st.text_input("Ask your question")

if user_input:
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, X)
    index = similarity.argmax()
    st.success(answers[index])