import streamlit as st
from chatbot_backened import GitaChatbot

st.title("📖 Bhagavad Gita Chatbot")

if "bot" not in st.session_state:
    st.session_state.bot = GitaChatbot()

query = st.text_input("Ask something about Bhagavad Gita:")

if query:
    verses, explanation = st.session_state.bot.chat(query)
    if verses:
        for v in verses:
            st.write("📜 Verse:", v)
    st.success(explanation)

