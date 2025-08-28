import streamlit as st
import os

st.set_page_config(page_title="Playlist Pal", layout="wide")

st.title("Playlist Pal - AI Music Curation")
st.markdown("Welcome to Playlist Pal! This is a working version.")

# Check API status
google_key = os.getenv("GOOGLE_API_KEY")
if google_key:
    st.success("Google API Connected")
else:
    st.error("Google API Key Missing")

st.markdown("## Features")
st.markdown("- Chain of Thought Prompting")
st.markdown("- Text Embeddings")
st.markdown("- Cosine Similarity")
st.markdown("- AI Playlist Generation")

st.info("To run the full application, install dependencies and set up your .env file.")
