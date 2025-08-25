# 🎵 Playlist Pal: Your AI-Powered Spotify DJ

Playlist Pal is a Python application that uses the power of Large Language Models (LLMs) to create curated Spotify playlists based on your mood, a description, or any creative idea you can think of. Instead of just matching genres, Playlist Pal understands the *vibe* of your request to find the perfect tracks.

## ✨ Core Features

* **Natural Language Playlist Creation**: Simply describe the playlist you want (e.g., "A playlist for a rainy afternoon in a coffee shop" or "upbeat 80s rock for a workout").
* **Intelligent Song Curation**: Uses function calling to interact with the Spotify API, finding tracks that match the vibe.
* **Semantic Song Search**: Leverages embeddings and a vector database to find songs that are conceptually similar to your request, going beyond simple keyword matching.
* **Customizable AI**: Demonstrates the use of various LLM parameters like Temperature, Top-P, and Top-K to control the creativity of the AI's suggestions.

## 🛠️ Tech Stack

* **Language**: Python 3.9+
* **LLM**: Google Gemini API
* **Embeddings**: Google's `text-embedding-004` model
* **Vector Database**: ChromaDB (for local semantic search)
* **Spotify Integration**: Spotipy (Python library for the Spotify Web API)

## 🚀 Getting Started

### Prerequisites

1.  **Python**: Make sure you have Python 3.9 or newer installed.
2.  **Google AI API Key**: Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3.  **Spotify API Credentials**:
    * Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
    * Create an app to get your `Client ID` and `Client Secret`.
    * In your app settings, set a "Redirect URI" to `http://localhost:8888/callback`.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Your `requirements.txt` file should include `google-generativeai`, `spotipy`, `chromadb`, `numpy`, `python-dotenv`)*

3.  **Set up environment variables:**
    Create a file named `.env` in the root directory and add your credentials:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    SPOTIPY_CLIENT_ID="YOUR_SPOTIFY_CLIENT_ID"
    SPOTIPY_CLIENT_SECRET="YOUR_SPOTIFY_CLIENT_SECRET"
    SPOTIPY_REDIRECT_URI="http://localhost:8888/callback"
    ```

### Running the Project

Execute the main Python script to start the application:
```bash
python main.py