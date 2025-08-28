#!/usr/bin/env python3
"""
start_app.py - Quick Start Script for Playlist Pal Pro

This script helps you quickly start the appropriate version of the application.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def check_environment():
    """Check if environment is properly set up"""
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    spotify_id = os.getenv("SPOTIPY_CLIENT_ID")
    spotify_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    
    print("Ì¥ç Checking Environment Setup...")
    print(f"  Google API Key: {'‚úÖ Found' if google_key else '‚ùå Missing'}")
    print(f"  Spotify Client ID: {'‚úÖ Found' if spotify_id else '‚ùå Missing'}")
    print(f"  Spotify Client Secret: {'‚úÖ Found' if spotify_secret else '‚ùå Missing'}")
    
    return bool(google_key and spotify_id and spotify_secret)

def main():
    print("Ìæµ Playlist Pal Pro - Quick Start")
    print("=" * 40)
    
    if not check_environment():
        print("\n‚ö†Ô∏è  Environment not fully configured!")
        print("Please set up your .env file with API keys.")
        print("See SETUP_GUIDE.md for instructions.")
        return
    
    print("\nÔøΩÔøΩ Choose your application version:")
    print("1. Ìæµ Full Version (with Spotify integration)")
    print("2. ÌæØ Basic Version (AI demos only)")
    print("3. Ì≥ä Analytics Dashboard")
    print("4. Ì∂•Ô∏è  Command Line Demo")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\nÌ∫Ä Starting Full Version with Spotify Integration...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_with_spotify.py"])
    elif choice == "2":
        print("\nÌ∫Ä Starting Basic Version...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif choice == "3":
        print("\nÌ∫Ä Starting Analytics Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "analytics_engine.py"])
    elif choice == "4":
        print("\nÌ∫Ä Starting Command Line Demo...")
        subprocess.run([sys.executable, "demo.py"])
    else:
        print("‚ùå Invalid choice!")

if __name__ == "__main__":
    main()
