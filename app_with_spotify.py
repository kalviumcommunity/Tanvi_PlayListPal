import streamlit as st
import os

st.set_page_config(page_title="Playlist Pal Pro", layout="wide")

st.title("Playlist Pal Pro - Enhanced Version")
st.markdown("Enhanced version with Spotify integration capabilities.")

# Check API status
google_key = os.getenv("GOOGLE_API_KEY")
spotify_id = os.getenv("SPOTIPY_CLIENT_ID")

col1, col2 = st.columns(2)
with col1:
    if google_key:
        st.success("Google API Connected")
    else:
        st.error("Google API Key Missing")

with col2:
    if spotify_id:
        st.success("Spotify Credentials Found")
    else:
        st.warning("Spotify credentials not configured")

st.markdown("## Enhanced Features")
st.markdown("- Chain of Thought Prompting")
st.markdown("- Text Embeddings & Vector Search")
st.markdown("- Cosine Similarity Analysis")
st.markdown("- AI Playlist Generation")
st.markdown("- Spotify Integration (when configured)")
st.markdown("- Analytics Dashboard")

st.info("Configure your .env file with API keys to unlock all features.")
