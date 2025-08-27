# main.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_playlist_with_cot(user_prompt: str):
    """
    Generates playlist details using a Chain of Thought prompt.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')


    ## CONCEPT: Chain of Thought Prompting 
    cot_prompt = f"""
    A user wants a playlist based on this request: "{user_prompt}"

    Let's think step by step to create the perfect playlist details:
    1.  First, what is the core mood or activity of the request? Identify the main keywords.
    2.  Second, what music genres or styles would fit this mood? Think of at least three.
    3.  Third, based on the mood and genres, come up with a creative and fitting playlist name.
    4.  Fourth, write a short, engaging description for the playlist (under 20 words).
    5.  Finally, combine the genres and keywords from the previous steps into a list of search terms.

    Based on this reasoning, provide the final output ONLY as a valid JSON object with the keys:
    'playlist_name', 'playlist_description', and 'search_terms'.
    """

    
    try:
        response = model.generate_content(
            cot_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.8
            )
        )
        print(f"Tokens used: {response.usage_metadata.total_token_count}")
        return json.loads(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    print("🎵 Welcome to Playlist Pal (Chain of Thought Demo) 🎵")
    idea = input("What kind of playlist are you in the mood for today?\n> ")
    
    if idea:
        playlist_data = get_playlist_with_cot(idea)
        
        if playlist_data:
            print("\n--- AI Generated Playlist ---")
            print(json.dumps(playlist_data, indent=2))
            print("---------------------------")