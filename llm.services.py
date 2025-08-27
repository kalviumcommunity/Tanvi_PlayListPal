import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_playlist_details(user_prompt: str):
    """
    Uses Gemini to generate playlist details based on a user's prompt.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

  
    ## CONCEPT 2: 
    system_prompt = """
    You are a world-class music expert and a brilliant Spotify playlist curator.
    Your task is to analyze a user's request and generate a fitting playlist name,
    a short, vibrant description (under 20 words), and a list of 3-5 relevant
    search terms (genres, moods, artists) that can be used to find songs.

    You must reply ONLY with a valid JSON object, without any surrounding text or markdown.
    The JSON object must have these three keys: 'playlist_name', 'playlist_description', 'search_terms'.
    """
    
    full_prompt = f"{system_prompt}\n\nUser request: \"{user_prompt}\""
  