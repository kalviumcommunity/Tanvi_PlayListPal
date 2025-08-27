

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
  

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                
                ## CONCEPT 3: Structured Output ##
            
                response_mime_type="application/json",
                

                
                ## CONCEPT 4: Temperature ##

                
                temperature=0.8
                
            )
        )

        
        ## CONCEPT 5: 
        
        
        print(f"Tokens used: {response.usage_metadata.total_token_count}")
        

        return json.loads(response.text)

    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return None


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