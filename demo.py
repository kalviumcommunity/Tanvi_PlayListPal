#!/usr/bin/env python3
"""
demo.py - Interactive Demo Script for Playlist Pal Pro

This script provides a command-line interface to test all the core features
of Playlist Pal without requiring the web interface.
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def check_environment():
    """Check if all required environment variables are set"""
    print("Ì¥ç Checking Environment Setup...")
    
    required_vars = {
        'GOOGLE_API_KEY': 'Google AI API Key',
        'SPOTIPY_CLIENT_ID': 'Spotify Client ID',
        'SPOTIPY_CLIENT_SECRET': 'Spotify Client Secret'
    }
    
    all_good = True
    for var, description in required_vars.items():
        if os.getenv(var):
            print(f"  ‚úÖ {description}: Found")
        else:
            print(f"  ‚ùå {description}: Missing")
            all_good = False
    
    if all_good:
        print("Ìæâ Environment setup complete!")
    else:
        print("‚ö†Ô∏è  Please check your .env file")
    
    return all_good

def demo_chain_of_thought():
    """Demonstrate Chain of Thought prompting"""
    print("\nÌæØ Chain of Thought Prompting Demo")
    print("-" * 40)
    
    try:
        from main import get_playlist_with_cot
        
        prompt = input("Describe your playlist idea: ")
        if prompt:
            print("Ì∑† AI is thinking step by step...")
            result = get_playlist_with_cot(prompt)
            
            if result:
                print("\n‚úÖ Generated Playlist Concept:")
                print(json.dumps(result, indent=2))
            else:
                print("‚ùå Failed to generate playlist")
        
    except Exception as e:
        print(f"‚ùå Error: {e}")

def demo_embeddings():
    """Demonstrate text embeddings"""
    print("\nÌ∑† Text Embeddings Demo")
    print("-" * 40)
    
    try:
        text = input("Enter text to convert to embedding: ")
        if text:
            print("Ì¥Ñ Generating embedding...")
            
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            embedding = result['embedding']
            print(f"‚úÖ Generated {len(embedding)}-dimensional embedding!")
            print(f"First 10 values: {embedding[:10]}")
            
            return embedding
        
    except Exception as e:
        print(f"‚ùå Error: {e}")
        return None

def demo_cosine_similarity():
    """Demonstrate cosine similarity calculation"""
    print("\nÌ≥ä Cosine Similarity Demo")
    print("-" * 40)
    
    try:
        from cosine import cosine_similarity
        
        # Preset vectors for demo
        vectors = {
            "Workout Music": [0.1, 0.8, 0.2, 0.3, 0.7, 0.1],
            "Chill Music": [0.8, 0.2, 0.9, 0.7, 0.1, 0.8],
            "Electronic": [0.2, 0.9, 0.1, 0.2, 0.8, 0.3],
            "Classical": [0.9, 0.1, 0.8, 0.9, 0.2, 0.7]
        }
        
        print("Available vectors:")
        for name in vectors.keys():
            print(f"  - {name}")
        
        vec1_name = input("Choose first vector: ")
        vec2_name = input("Choose second vector: ")
        
        if vec1_name in vectors and vec2_name in vectors:
            vec1 = vectors[vec1_name]
            vec2 = vectors[vec2_name]
            
            similarity = cosine_similarity(vec1, vec2)
            print(f"\nÌ≥ä Cosine Similarity: {similarity:.4f}")
            
            if similarity > 0.8:
                print("ÔøΩÔøΩ Very similar!")
            elif similarity > 0.5:
                print("Ì¥î Moderately similar")
            elif similarity > 0.2:
                print("Ì≥ä Somewhat similar")
            else:
                print("‚ùå Not very similar")
        else:
            print("‚ùå Invalid vector names")
        
    except Exception as e:
        print(f"‚ùå Error: {e}")

def main():
    """Main demo function"""
    print("Ìæµ Welcome to Playlist Pal Pro Demo! Ìæµ")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n‚ö†Ô∏è  Please set up your environment variables first!")
        print("See SETUP_GUIDE.md for instructions.")
        return
    
    while True:
        print("\nÌæõÔ∏è  Choose a demo:")
        print("1. ÌæØ Chain of Thought Prompting")
        print("2. Ì∑† Text Embeddings")
        print("3. Ì≥ä Cosine Similarity")
        print("4. Ì∫Ä Run Full Web App")
        print("0. ‚ùå Exit")
        
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == "1":
            demo_chain_of_thought()
        elif choice == "2":
            demo_embeddings()
        elif choice == "3":
            demo_cosine_similarity()
        elif choice == "4":
            print("\nÌ∫Ä Launching web application...")
            print("Run: streamlit run app_with_spotify.py")
            break
        elif choice == "0":
            print("\nÌ±ã Thanks for trying Playlist Pal Pro!")
            break
        else:
            print("‚ùå Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
