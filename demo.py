# -*- coding: utf-8 -*-
"""
demo.py - Simple Demo Script
"""

import os
from dotenv import load_dotenv

def main():
    print("Playlist Pal Pro Demo")
    print("=" * 30)
    
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    spotify_id = os.getenv("SPOTIPY_CLIENT_ID")
    
    print(f"Google API Key: {\"Found\" if google_key else \"Missing\"}")
    print(f"Spotify Client ID: {\"Found\" if spotify_id else \"Missing\"}")
    
    print("\nTo run the full application:")
    print("streamlit run app_with_spotify.py")

if __name__ == "__main__":
    main()
