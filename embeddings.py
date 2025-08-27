import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_and_show_embedding(text: str):
    """Generates an embedding and displays its properties."""
    print(f"--- Generating Embedding for: '{text}' ---")
    
    try:
        
        ## CONCEPT: Embeddings ##
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT" 
        )
        embedding_vector = result['embedding']
        
        
        print(f"✅ Successfully generated a {len(embedding_vector)}-dimensional vector.")
        print(f"Here are the first 5 numbers of the vector: {embedding_vector[:5]}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    idea = input("Enter a phrase to turn into an embedding (e.g., 'Music for a sunny day in Jaipur'):\n> ")
    if idea:
        generate_and_show_embedding(idea)