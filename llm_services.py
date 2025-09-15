

import os
import json
import time
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def _parse_json_strict_or_fallback(text: str):
    """
    Try strict JSON parse; on failure, extract first {...} block and parse.
    """
    if not text:
        raise ValueError("Empty response text")
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Fallback: extract first JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            return json.loads(candidate)
        raise

def _extract_retry_delay_seconds(error_text: str, default_seconds: int = 30) -> int:
    # Try to find 'retry_delay { seconds: N }' or 'Retry-After: N'
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", error_text)
    if m:
        return int(m.group(1))
    m = re.search(r"Retry-After:\s*(\d+)", error_text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return default_seconds

def get_playlist_details_with_error(user_prompt: str):
    """
    Same as get_playlist_details but returns a tuple: (data_dict_or_none, error_str_or_none)
    Includes a single retry if a 429 quota error occurs.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    system_prompt = """
    You are a world-class music expert and a brilliant Spotify playlist curator.
    Your task is to analyze a user's request and generate a fitting playlist name,
    a short, vibrant description (under 20 words), and a list of 3-5 relevant
    search terms (genres, moods, artists) that can be used to find songs.

    You must reply ONLY with a valid JSON object, without any surrounding text or markdown.
    The JSON object must have these three keys: 'playlist_name', 'playlist_description', 'search_terms'.
    """
    full_prompt = f"{system_prompt}\n\nUser request: \"{user_prompt}\""

    def _attempt():
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.8
            )
        )
        try:
            print(f"Tokens used: {getattr(response, 'usage_metadata', {}).get('total_token_count', 'n/a')}")
        except Exception:
            pass
        data = _parse_json_strict_or_fallback(response.text)
        return data

    # First attempt
    try:
        return _attempt(), None
    except Exception as e:
        err_text = str(e)
        if "429" in err_text or "quota" in err_text.lower():
            # Single retry honoring suggested delay if present
            delay = _extract_retry_delay_seconds(err_text, default_seconds=30)
            try:
                time.sleep(min(delay, 60))
                return _attempt(), None
            except Exception as e2:
                return None, str(e2)
        return None, err_text

def get_playlist_details(user_prompt: str):
    """
    Uses Gemini to generate playlist details based on a user's prompt.
    Returns a dict with keys: playlist_name, playlist_description, search_terms.
    """
    data, err = get_playlist_details_with_error(user_prompt)
    if err:
        print(f"An error occurred while calling the Gemini API: {err}")
        return None
    return data


if __name__ == '__main__':
    #  input
    idea = input("What kind of playlist are you in the mood for?\n> ")
    
    if idea:
        # Call 
        playlist_data = get_playlist_details(idea)
        
        if playlist_data:
            print("\n--- AI Generated Playlist ---")
            # print
            print(json.dumps(playlist_data, indent=2))
            print("---------------------------")