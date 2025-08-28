# -*- coding: utf-8 -*-
"""
analytics_engine.py - Simple Analytics Dashboard
"""

import streamlit as st
import json
from datetime import datetime

def main():
    st.title("Playlist Pal Analytics Dashboard")
    
    st.markdown("## Usage Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", "1,247")
    with col2:
        st.metric("Playlist Generated", "1,198")
    with col3:
        st.metric("Success Rate", "96.1%")
    with col4:
        st.metric("Avg Response Time", "2.3s")
    
    st.markdown("## AI Model Performance")
    
    # Sample data
    performance_data = {
        "Chain of Thought": 95.2,
        "Embeddings": 98.7,
        "Similarity": 97.1,
        "Spotify Integration": 94.8
    }
    
    for feature, score in performance_data.items():
        st.progress(score / 100)
        st.write(f"{feature}: {score}%")
    
    st.markdown("## Recent Activity")
    
    # Sample recent activity
    activities = [
        "User generated workout playlist",
        "Embedding calculation completed",
        "Spotify playlist created",
        "Similarity analysis performed"
    ]
    
    for activity in activities:
        st.write(f"• {activity}")

if __name__ == "__main__":
    main()
