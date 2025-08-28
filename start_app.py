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
    
    print("� Checking Environment Setup...")
    print(f"  Google API Key: {'✅ Found' if google_key else '❌ Missing'}")
    print(f"  Spotify Client ID: {'✅ Found' if spotify_id else '❌ Missing'}")
    print(f"  Spotify Client Secret: {'✅ Found' if spotify_secret else '❌ Missing'}")
    
    return bool(google_key and spotify_id and spotify_secret)

def main():
    print("� Playlist Pal Pro - Quick Start")
    print("=" * 40)
    
    if not check_environment():
        print("\n⚠️  Environment not fully configured!")
        print("Please set up your .env file with API keys.")
        print("See SETUP_GUIDE.md for instructions.")
        return
    
    print("\n�� Choose your application version:")
    print("1. � Full Version (with Spotify integration)")
    print("2. � Basic Version (AI demos only)")
    print("3. � Analytics Dashboard")
    print("4. �️  Command Line Demo")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n� Starting Full Version with Spotify Integration...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_with_spotify.py"])
    elif choice == "2":
        print("\n� Starting Basic Version...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif choice == "3":
        print("\n� Starting Analytics Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "analytics_engine.py"])
    elif choice == "4":
        print("\n� Starting Command Line Demo...")
        subprocess.run([sys.executable, "demo.py"])
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
