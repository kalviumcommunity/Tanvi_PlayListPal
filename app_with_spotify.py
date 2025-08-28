# app_with_spotify.py - Enhanced Streamlit Application with Spotify Integration

import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np

# Import our existing modules
from llm_services import get_playlist_details
from embeddings import generate_and_show_embedding
from cosine import cosine_similarity
from spotify_service import SpotifyPlaylistManager, create_complete_playlist_from_ai, get_spotify_auth_url
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Ìæµ Playlist Pal Pro",
    page_icon="Ìæµ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1DB954;
        margin: 1rem 0;
    }
    .concept-badge {
        background: #1DB954;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .similarity-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1DB954;
    }
    .spotify-card {
        background: #191414;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .track-item {
        background: #f0f0f0;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        border-left: 3px solid #1DB954;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize Spotify manager in session state
    if 'spotify_manager' not in st.session_state:
        st.session_state.spotify_manager = SpotifyPlaylistManager()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Ìæµ Playlist Pal Pro: AI + Spotify Integration</h1>
        <p>Create real Spotify playlists using AI and machine learning!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation and Spotify status
    with st.sidebar:
        st.title("ÌæõÔ∏è Controls")
        
        # Spotify Authentication Status
        st.markdown("### Ìæµ Spotify Connection")
        spotify_manager = st.session_state.spotify_manager
        
        if spotify_manager.is_authenticated():
            user_info = spotify_manager.get_user_info()
            if user_info:
                st.success(f"‚úÖ Connected as: {user_info.get('display_name', 'User')}")
                st.image(user_info.get('images', [{}])[0].get('url', ''), width=60)
            else:
                st.success("‚úÖ Spotify Connected")
        else:
            st.error("‚ùå Not connected to Spotify")
            if st.button("Ì¥ó Connect to Spotify"):
                auth_url = get_spotify_auth_url()
                st.markdown(f"[Click here to authenticate with Spotify]({auth_url})")
                st.info("After authentication, please restart the app.")
        
        st.markdown("---")
        
        # Page selection
        page = st.selectbox(
            "Choose Feature",
            ["ÔøΩÔøΩ Home", "ÌæØ Chain of Thought", "Ì∑† Embeddings Demo", "Ì≥ä Cosine Similarity", 
             "Ìæµ AI Playlist Generator", "Ìæß Spotify Integration", "Ì≥à Analytics Dashboard"]
        )
        
        st.markdown("---")
        st.markdown("### Ì¥ë API Status")
        
        # Check API key status
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            st.success("‚úÖ Google AI Connected")
        else:
            st.error("‚ùå Google API Key Missing")

    # Main content routing
    if page == "Ìø† Home":
        show_home_page()
    elif page == "ÌæØ Chain of Thought":
        show_chain_of_thought()
    elif page == "Ì∑† Embeddings Demo":
        show_embeddings_demo()
    elif page == "Ì≥ä Cosine Similarity":
        show_cosine_similarity()
    elif page == "Ìæµ AI Playlist Generator":
        show_ai_playlist_generator()
    elif page == "Ìæß Spotify Integration":
        show_spotify_integration()
    elif page == "Ì≥à Analytics Dashboard":
        show_analytics_dashboard()

def show_home_page():
    st.markdown("## Ìºü Welcome to Playlist Pal Pro")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>ÌæØ Enhanced Features</h3>
            <ul>
                <li><strong>Ìæµ Real Spotify Integration</strong> - Create actual playlists</li>
                <li><strong>Ì∑† AI-Powered Curation</strong> - Smart track selection</li>
                <li><strong>Ì≥ä Advanced Analytics</strong> - Detailed playlist insights</li>
                <li><strong>ÌæØ Similarity Analysis</strong> - Audio feature matching</li>
                <li><strong>Ì¥ç Smart Search</strong> - Multi-term track discovery</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Ì∫Ä AI/ML Concepts Demonstrated</h3>
            <ul>
                <li><strong>Chain of Thought Prompting</strong></li>
                <li><strong>Text Embeddings & Vector Search</strong></li>
                <li><strong>Cosine Similarity</strong></li>
                <li><strong>Function Calling & API Integration</strong></li>
                <li><strong>Structured Output Generation</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    spotify_manager = st.session_state.spotify_manager
    if spotify_manager.is_authenticated():
        playlists = spotify_manager.get_user_playlists()
        playlist_count = len(playlists)
    else:
        playlist_count = "N/A"
    
    with col1:
        st.metric("Ìæµ Your Playlists", playlist_count)
    with col2:
        st.metric("Ì∑† AI Model", "Gemini 1.5")
    with col3:
        st.metric("Ì≥ä Embedding Dims", "768")
    with col4:
        st.metric("ÌæØ Features", "7")

def show_chain_of_thought():
    st.markdown("## ÌæØ Chain of Thought Prompting")
    
    st.markdown("""
    <div class="concept-badge">Concept: Chain of Thought (CoT) Prompting</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Chain of Thought prompting** breaks down complex reasoning into clear steps,
    improving the quality and explainability of AI responses.
    """)
    
    # Interactive demo
    user_input = st.text_input(
        "Describe your ideal playlist:",
        placeholder="e.g., Energetic workout music with heavy bass and electronic beats",
        help="The AI will reason through your request step by step."
    )
    
    if st.button("ÌæØ Generate with Chain of Thought", type="primary"):
        if user_input:
            with st.spinner("Ì∑† AI is reasoning step by step..."):
                from main import get_playlist_with_cot
                result = get_playlist_with_cot(user_input)
                
                if result:
                    st.success("‚úÖ Playlist concept generated!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Ìæµ Generated Concept")
                        st.json(result)
                    
                    with col2:
                        st.markdown("### Ì¥ç Analysis")
                        st.write(f"**Name:** {result.get('playlist_name', 'N/A')}")
                        st.write(f"**Description:** {result.get('playlist_description', 'N/A')}")
                        
                        if 'search_terms' in result:
                            st.write("**Search Terms:**")
                            for term in result['search_terms']:
                                st.markdown(f"- `{term}`")
                        
                        # Option to create Spotify playlist
                        if st.session_state.spotify_manager.is_authenticated():
                            if st.button("Ìæµ Create Spotify Playlist"):
                                spotify_result = create_complete_playlist_from_ai(
                                    st.session_state.spotify_manager, result, max_tracks=25
                                )
                                if spotify_result:
                                    st.success(f"Ìæâ Created playlist: {spotify_result['playlist_name']}")
                                    st.markdown(f"[Open in Spotify]({spotify_result['spotify_url']})")
                else:
                    st.error("‚ùå Failed to generate playlist. Check your API key.")
        else:
            st.warning("‚ö†Ô∏è Please enter a playlist description.")

def show_embeddings_demo():
    st.markdown("## Ì∑† Embeddings Demo")
    
    st.markdown("""
    <div class="concept-badge">Concept: Text Embeddings</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Embeddings** convert text into numerical vectors that capture semantic meaning.
    This enables similarity calculations and vector-based search.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ÌæÆ Generate Embedding")
        text_input = st.text_area(
            "Enter music description:",
            placeholder="e.g., upbeat dance music for parties",
            height=100
        )
        
        if st.button("Ì∑† Generate Embedding", type="primary"):
            if text_input:
                with st.spinner("Ì¥Ñ Generating embedding..."):
                    try:
                        result = genai.embed_content(
                            model="models/text-embedding-004",
                            content=text_input,
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                        embedding = result['embedding']
                        
                        st.success(f"‚úÖ Generated {len(embedding)}-dimensional embedding!")
                        
                        # Store in session state
                        if 'embeddings' not in st.session_state:
                            st.session_state.embeddings = []
                        
                        st.session_state.embeddings.append({
                            'text': text_input,
                            'embedding': embedding,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Show embedding visualization
                        fig = px.line(
                            y=embedding[:50], 
                            title="First 50 Dimensions",
                            labels={'index': 'Dimension', 'y': 'Value'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Ì≥ä Mean", f"{np.mean(embedding):.4f}")
                        with col_b:
                            st.metric("Ì≥à Max", f"{max(embedding):.4f}")
                        with col_c:
                            st.metric("Ì≥â Min", f"{min(embedding):.4f}")
                        
                    except Exception as e:
                        st.error(f"‚ùå Error: {e}")
            else:
                st.warning("‚ö†Ô∏è Please enter some text.")
    
    with col2:
        st.markdown("### Ì≥ä Embedding History")
        if 'embeddings' in st.session_state and st.session_state.embeddings:
            for i, emb in enumerate(st.session_state.embeddings[-5:]):
                with st.expander(f"Ìµê {emb['timestamp']} - {emb['text'][:30]}..."):
                    st.write(f"**Full text:** {emb['text']}")
                    st.write(f"**Dimensions:** {len(emb['embedding'])}")
                    
                    # Compare with others
                    if len(st.session_state.embeddings) > 1:
                        similarities = []
                        for other in st.session_state.embeddings:
                            if other != emb:
                                sim = cosine_similarity(emb['embedding'], other['embedding'])
                                similarities.append((other['text'][:20] + "...", sim))
                        
                        if similarities:
                            st.write("**Most similar:**")
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            for text, sim in similarities[:3]:
                                st.write(f"- {text}: {sim:.3f}")
        else:
            st.info("Ì≤° Generate embeddings to see history here!")

def show_cosine_similarity():
    st.markdown("## Ì≥ä Cosine Similarity Calculator")
    
    st.markdown("""
    <div class="concept-badge">Concept: Cosine Similarity</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Cosine similarity** measures the angle between vectors, ranging from -1 to 1.
    It's perfect for comparing the semantic similarity of text embeddings.
    """)
    
    # Music genre similarity demo
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ÌæÆ Music Genre Similarity")
        
        genre_vectors = {
            "Electronic Dance": [0.9, 0.8, 0.1, 0.2, 0.9, 0.7],
            "Classical": [0.1, 0.2, 0.9, 0.8, 0.1, 0.3],
            "Rock": [0.7, 0.9, 0.3, 0.1, 0.8, 0.6],
            "Jazz": [0.4, 0.5, 0.7, 0.6, 0.4, 0.5],
            "Hip Hop": [0.8, 0.7, 0.2, 0.3, 0.9, 0.8],
            "Ambient": [0.2, 0.1, 0.8, 0.9, 0.2, 0.3]
        }
        
        genre1 = st.selectbox("First genre:", list(genre_vectors.keys()))
        genre2 = st.selectbox("Second genre:", list(genre_vectors.keys()))
        
        if st.button("Ì≥ä Calculate Similarity", type="primary"):
            vec1 = genre_vectors[genre1]
            vec2 = genre_vectors[genre2]
            
            similarity = cosine_similarity(vec1, vec2)
            
            # Display result
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <h3>Similarity Score</h3>
                <div class="similarity-score">{similarity:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation
            if similarity > 0.8:
                st.success("ÌæØ Very Similar! These genres share many characteristics.")
            elif similarity > 0.5:
                st.info("Ì¥î Moderately Similar. Some overlap in style.")
            elif similarity > 0.2:
                st.warning("Ì≥ä Somewhat Similar. Limited relationship.")
            else:
                st.error("‚ùå Very Different. Distinct musical styles.")
            
            # Visualize comparison
            fig = go.Figure()
            dimensions = ["Energy", "Beat", "Melody", "Harmony", "Rhythm", "Intensity"]
            
            fig.add_trace(go.Scatterpolar(
                r=vec1,
                theta=dimensions,
                fill='toself',
                name=genre1,
                line_color='#1DB954'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=vec2,
                theta=dimensions,
                fill='toself',
                name=genre2,
                line_color='#ff6b6b'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                title="Genre Characteristic Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Ì∑Æ Custom Vector Calculator")
        
        st.markdown("Create your own vectors:")
        custom_vec1 = st.text_input("Vector 1 (comma-separated):", placeholder="0.8, 0.6, 0.4, 0.2")
        custom_vec2 = st.text_input("Vector 2 (comma-separated):", placeholder="0.7, 0.5, 0.3, 0.1")
        
        if st.button("Ì¥¢ Calculate Custom Similarity"):
            try:
                vec1 = [float(x.strip()) for x in custom_vec1.split(',')]
                vec2 = [float(x.strip()) for x in custom_vec2.split(',')]
                
                if len(vec1) != len(vec2):
                    st.error("‚ùå Vectors must have the same length!")
                else:
                    similarity = cosine_similarity(vec1, vec2)
                    st.success(f"‚úÖ Similarity Score: **{similarity:.4f}**")
                    
                    # Show calculation steps
                    with st.expander("Ì≥ã Calculation Details"):
                        dot_product = np.dot(vec1, vec2)
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        
                        st.write(f"**Dot Product:** {dot_product:.4f}")
                        st.write(f"**||Vector 1||:** {norm1:.4f}")
                        st.write(f"**||Vector 2||:** {norm2:.4f}")
                        st.write(f"**Cosine Similarity:** {similarity:.4f}")
                        
            except ValueError:
                st.error("‚ùå Please enter valid numbers separated by commas.")

def show_ai_playlist_generator():
    st.markdown("## Ìæµ AI Playlist Generator")
    
    st.markdown("""
    <div class="concept-badge">Concept: Integrated AI Pipeline</div>
    """, unsafe_allow_html=True)
    
    # Configuration
    with st.expander("‚öôÔ∏è Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.slider("Ìº°Ô∏è Creativity", 0.0, 1.0, 0.8, 0.1)
        with col2:
            max_tracks = st.slider("Ìæµ Max Tracks", 10, 50, 25)
        with col3:
            search_depth = st.slider("Ì¥ç Search Depth", 10, 50, 20)
    
    # Main input
    playlist_prompt = st.text_area(
        "Describe your perfect playlist:",
        placeholder="Examples:\n- Chill lo-fi beats for studying\n- High-energy gym workout music\n- Nostalgic 2000s pop hits\n- Ambient music for meditation",
        height=120
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Ìæµ Generate AI Playlist", type="primary"):
            if playlist_prompt:
                generate_ai_playlist(playlist_prompt, temperature, max_tracks, search_depth)
            else:
                st.warning("‚ö†Ô∏è Please describe your playlist!")
    
    with col2:
        if st.button("Ì¥Ñ Clear Results"):
            if 'ai_playlist_results' in st.session_state:
                del st.session_state.ai_playlist_results

def generate_ai_playlist(prompt, temperature, max_tracks, search_depth):
    """Generate a complete AI playlist with analysis"""
    
    with st.spinner("Ì¥ñ Generating your AI playlist..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate playlist concept
            status_text.text("ÌæØ Step 1/4: Creating playlist concept...")
            progress_bar.progress(25)
            
            # Use the enhanced LLM service with custom temperature
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            enhanced_prompt = f"""
            Create a detailed playlist concept for: "{prompt}"
            
            Provide a JSON response with:
            - playlist_name: Creative, fitting name
            - playlist_description: Engaging description (under 30 words)  
            - search_terms: 5-7 specific search terms for finding tracks
            - mood_tags: 3-5 mood/genre tags
            - target_audience: Who would enjoy this playlist
            """
            
            response = model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=temperature
                )
            )
            
            playlist_concept = json.loads(response.text)
            
            # Step 2: Generate embeddings
            status_text.text("Ì∑† Step 2/4: Creating semantic embeddings...")
            progress_bar.progress(50)
            
            embedding_result = genai.embed_content(
                model="models/text-embedding-004",
                content=prompt,
                task_type="RETRIEVAL_DOCUMENT"
            )
            prompt_embedding = embedding_result['embedding']
            
            # Step 3: Simulate track search (would use Spotify in full version)
            status_text.text("Ì¥ç Step 3/4: Searching for matching tracks...")
            progress_bar.progress(75)
            
            # Simulate finding tracks based on search terms
            simulated_tracks = generate_simulated_tracks(
                playlist_concept.get('search_terms', []), 
                max_tracks
            )
            
            # Step 4: Analyze and finalize
            status_text.text("‚ú® Step 4/4: Analyzing playlist coherence...")
            progress_bar.progress(100)
            
            # Calculate genre similarities
            genre_analysis = analyze_genre_fit(prompt_embedding, playlist_concept.get('mood_tags', []))
            
            time.sleep(1)
            
            # Store results
            st.session_state.ai_playlist_results = {
                'prompt': prompt,
                'concept': playlist_concept,
                'embedding': prompt_embedding,
                'tracks': simulated_tracks,
                'genre_analysis': genre_analysis,
                'generation_settings': {
                    'temperature': temperature,
                    'max_tracks': max_tracks,
                    'search_depth': search_depth
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            display_ai_playlist_results()
            
        except Exception as e:
            st.error(f"‚ùå Error generating playlist: {e}")
            progress_bar.empty()
            status_text.empty()

def generate_simulated_tracks(search_terms, max_tracks):
    """Generate simulated track data for demo purposes"""
    import random
    
    # Sample track templates
    track_templates = [
        {"name": "Midnight Vibes", "artist": "Lo-Fi Dreams", "popularity": 78},
        {"name": "Electric Pulse", "artist": "Synth Masters", "popularity": 85},
        {"name": "Chill Waves", "artist": "Ambient Collective", "popularity": 72},
        {"name": "Beat Drop", "artist": "Bass Heavy", "popularity": 91},
        {"name": "Nostalgic Flow", "artist": "Retro Sounds", "popularity": 68},
        {"name": "Study Session", "artist": "Focus Music", "popularity": 75},
        {"name": "Workout Energy", "artist": "Gym Beats", "popularity": 88},
        {"name": "Dreamy Nights", "artist": "Sleep Sounds", "popularity": 70}
    ]
    
    tracks = []
    for i in range(min(max_tracks, len(search_terms) * 4)):
        template = random.choice(track_templates)
        search_term = random.choice(search_terms) if search_terms else "music"
        
        track = {
            'id': f"track_{i}",
            'name': f"{template['name']} ({search_term.title()})",
            'artist': template['artist'],
            'album': f"Album {i+1}",
            'duration_ms': random.randint(180000, 300000),  # 3-5 minutes
            'popularity': template['popularity'] + random.randint(-10, 10),
            'search_term': search_term
        }
        tracks.append(track)
    
    return tracks

def analyze_genre_fit(embedding, mood_tags):
    """Analyze how well the prompt fits different genres"""
    
    # Simulate genre embeddings (in real app, these would be pre-computed)
    genre_vectors = {
        "Electronic": np.random.normal(0.3, 0.2, 768),
        "Pop": np.random.normal(0.5, 0.2, 768),
        "Rock": np.random.normal(-0.1, 0.3, 768),
        "Hip-Hop": np.random.normal(0.2, 0.25, 768),
        "Jazz": np.random.normal(-0.2, 0.3, 768),
        "Classical": np.random.normal(-0.3, 0.2, 768),
        "Ambient": np.random.normal(0.1, 0.15, 768)
    }
    
    similarities = {}
    for genre, genre_vec in genre_vectors.items():
        # Use only first 100 dimensions for speed in demo
        sim = cosine_similarity(embedding[:100], genre_vec[:100])
        similarities[genre] = max(0, sim)  # Ensure non-negative for visualization
    
    return similarities

def display_ai_playlist_results():
    """Display comprehensive AI playlist results"""
    
    if 'ai_playlist_results' not in st.session_state:
        return
    
    results = st.session_state.ai_playlist_results
    
    st.success("Ìæâ AI Playlist Generated Successfully!")
    
    # Main playlist info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Ìæµ Playlist Concept")
        concept = results['concept']
        
        st.markdown(f"""
        **Ìæº Name:** {concept.get('playlist_name', 'Untitled')}
        
        **Ì≥ù Description:** {concept.get('playlist_description', 'No description')}
        
        **ÌæØ Target Audience:** {concept.get('target_audience', 'Music lovers')}
        """)
        
        # Search terms
        if 'search_terms' in concept:
            st.markdown("**Ì¥ç Search Terms:**")
            for term in concept['search_terms']:
                st.markdown(f"- `{term}`")
        
        # Mood tags
        if 'mood_tags' in concept:
            st.markdown("**Ìø∑Ô∏è Mood Tags:**")
            tag_cols = st.columns(len(concept['mood_tags']))
            for i, tag in enumerate(concept['mood_tags']):
                with tag_cols[i]:
                    st.markdown(f"""
                    <div class="concept-badge">{tag}</div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ‚öôÔ∏è Generation Settings")
        settings = results['generation_settings']
        
        st.metric("Ìº°Ô∏è Temperature", f"{settings['temperature']:.1f}")
        st.metric("Ìæµ Track Count", len(results['tracks']))
        st.metric("Ì¥ç Search Depth", settings['search_depth'])
        st.metric("‚è∞ Generated", results['timestamp'].split()[1])
    
    # Genre analysis
    st.markdown("### Ì≥ä Genre Similarity Analysis")
    
    similarities = results['genre_analysis']
    genres = list(similarities.keys())
    scores = list(similarities.values())
    
    fig = px.bar(
        x=genres,
        y=scores,
        title="How well your playlist matches different genres",
        labels={'x': 'Genre', 'y': 'Similarity Score'},
        color=scores,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top matches
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    col1, col2, col3 = st.columns(3)
    
    for i, (genre, score) in enumerate(sorted_similarities[:3]):
        with [col1, col2, col3][i]:
            st.metric(f"#{i+1} {genre}", f"{score:.3f}")
    
    # Track listing
    st.markdown("### Ìæµ Track Listing")
    
    tracks_df = pd.DataFrame(results['tracks'])
    if not tracks_df.empty:
        # Display tracks in a nice format
        for i, track in enumerate(tracks_df.to_dict('records')[:10]):  # Show first 10
            duration_min = track['duration_ms'] // 60000
            duration_sec = (track['duration_ms'] % 60000) // 1000
            
            st.markdown(f"""
            <div class="track-item">
                <strong>{i+1}. {track['name']}</strong><br>
                <em>by {track['artist']}</em> ‚Ä¢ {track['album']} ‚Ä¢ {duration_min}:{duration_sec:02d}<br>
                <small>Popularity: {track['popularity']}/100 | Found via: "{track['search_term']}"</small>
            </div>
            """, unsafe_allow_html=True)
        
        if len(tracks_df) > 10:
            st.info(f"... and {len(tracks_df) - 10} more tracks")
    
    # Export options
    st.markdown("### Ì≤æ Export Options")
    
    export_data = {
        'playlist_concept': results['concept'],
        'tracks': results['tracks'],
        'genre_analysis': results['genre_analysis'],
        'generation_settings': results['generation_settings'],
        'prompt': results['prompt'],
        'generated_at': results['timestamp']
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Ì≥• Download Playlist (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"ai_playlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Option to create Spotify playlist if connected
        if st.session_state.spotify_manager.is_authenticated():
            if st.button("Ìæµ Create on Spotify"):
                st.info("Spotify playlist creation coming soon! (Need real Spotify integration)")

def show_spotify_integration():
    st.markdown("## Ìæß Spotify Integration")
    
    st.markdown("""
    <div class="concept-badge">Concept: API Integration & Function Calling</div>
    """, unsafe_allow_html=True)
    
    spotify_manager = st.session_state.spotify_manager
    
    if not spotify_manager.is_authenticated():
        st.warning("‚ö†Ô∏è Please connect to Spotify first!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="spotify-card">
                <h3>Ì¥ó Connect to Spotify</h3>
                <p>To use Spotify features, you need to:</p>
                <ol>
                    <li>Set up Spotify API credentials</li>
                    <li>Authenticate with your Spotify account</li>
                    <li>Grant necessary permissions</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.info("""
            **Required Environment Variables:**
            - `SPOTIPY_CLIENT_ID`
            - `SPOTIPY_CLIENT_SECRET`
            - `SPOTIPY_REDIRECT_URI`
            
            Get these from the Spotify Developer Dashboard.
            """)
        
        if st.button("Ì¥ó Get Auth URL"):
            try:
                auth_url = get_spotify_auth_url()
                st.markdown(f"[Ìæµ Click here to authenticate with Spotify]({auth_url})")
                st.info("After authentication, restart the app to see your connection.")
            except Exception as e:
                st.error(f"Error getting auth URL: {e}")
    
    else:
        # User is authenticated - show Spotify features
        user_info = spotify_manager.get_user_info()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if user_info:
                st.markdown(f"""
                <div class="spotify-card">
                    <h3>Ìæµ Connected User</h3>
                    <p><strong>{user_info.get('display_name', 'User')}</strong></p>
                    <p>Followers: {user_info.get('followers', {}).get('total', 0):,}</p>
                    <p>Country: {user_info.get('country', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Quick stats
            playlists = spotify_manager.get_user_playlists()
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Ìæµ Playlists", len(playlists))
            with col_b:
                public_count = sum(1 for p in playlists if p.get('public', False))
                st.metric("Ìºç Public", public_count)
            with col_c:
                total_tracks = sum(p.get('tracks', {}).get('total', 0) for p in playlists)
                st.metric("Ìæ∂ Total Tracks", total_tracks)
        
        # Feature tabs
        tab1, tab2, tab3 = st.tabs(["Ìæµ Your Playlists", "Ì¥ç Search Tracks", "Ì≥ä Audio Analysis"])
        
        with tab1:
            st.markdown("### Ìæµ Your Spotify Playlists")
            
            if playlists:
                for playlist in playlists[:10]:  # Show first 10
                    with st.expander(f"Ìæµ {playlist['name']} ({playlist['tracks']['total']} tracks)"):
                        col_left, col_right = st.columns([2, 1])
                        
                        with col_left:
                            st.write(f"**Description:** {playlist.get('description', 'No description')}")
                            st.write(f"**Owner:** {playlist['owner']['display_name']}")
                            st.write(f"**Public:** {'Yes' if playlist.get('public') else 'No'}")
                            
                            if playlist.get('external_urls', {}).get('spotify'):
                                st.markdown(f"[Ì¥ó Open in Spotify]({playlist['external_urls']['spotify']})")
                        
                        with col_right:
                            if playlist.get('images') and len(playlist['images']) > 0:
                                st.image(playlist['images'][0]['url'], width=100)
            else:
                st.info("No playlists found. Create some playlists in Spotify first!")
        
        with tab2:
            st.markdown("### Ì¥ç Search Spotify Tracks")
            
            search_query = st.text_input(
                "Search for tracks:",
                placeholder="e.g., artist:Taylor Swift, genre:pop, year:2023"
            )
            
            if st.button("Ì¥ç Search") and search_query:
                with st.spinner("Searching Spotify..."):
                    tracks = spotify_manager.search_tracks([search_query], limit=20)
                    
                    if tracks:
                        st.success(f"Found {len(tracks)} tracks!")
                        
                        for track in tracks[:5]:  # Show first 5
                            col_track, col_info = st.columns([3, 1])
                            
                            with col_track:
                                st.markdown(f"""
                                **{track['name']}** by {track['artist']}  
                                Album: {track['album']} | Popularity: {track['popularity']}/100
                                """)
                                
                                if track.get('external_urls', {}).get('spotify'):
                                    st.markdown(f"[Ìæµ Open in Spotify]({track['external_urls']['spotify']})")
                            
                            with col_info:
                                duration_min = track['duration_ms'] // 60000
                                duration_sec = (track['duration_ms'] % 60000) // 1000
                                st.write(f"‚è±Ô∏è {duration_min}:{duration_sec:02d}")
                    else:
                        st.warning("No tracks found for that search.")
        
        with tab3:
            st.markdown("### Ì≥ä Audio Feature Analysis")
            
            st.info("Select a playlist to analyze its audio features:")
            
            if playlists:
                selected_playlist = st.selectbox(
                    "Choose playlist:",
                    options=[p['name'] for p in playlists if p['tracks']['total'] > 0],
                    format_func=lambda x: f"{x} ({next(p['tracks']['total'] for p in playlists if p['name'] == x)} tracks)"
                )
                
                if st.button("Ì≥ä Analyze Audio Features"):
                    st.info("Audio feature analysis would be implemented here with real Spotify data.")
                    st.markdown("""
                    **Features that would be analyzed:**
                    - Danceability, Energy, Valence
                    - Acousticness, Speechiness, Liveness
                    - Tempo, Loudness, Key, Mode
                    - Similarity scores between tracks
                    """)

def show_analytics_dashboard():
    st.markdown("## Ì≥à Analytics Dashboard")
    
    st.markdown("""
    <div class="concept-badge">Concept: Data Visualization & Analytics</div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ÌæØ AI Model Performance")
    
    # Simulated analytics data
    model_stats = {
        'total_requests': 1247,
        'successful_generations': 1198,
        'avg_response_time': 2.3,
        'avg_temperature': 0.75,
        'most_popular_genre': 'Electronic'
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Ì≥ä Total Requests", f"{model_stats['total_requests']:,}")
    with col2:
        st.metric("‚úÖ Success Rate", f"{model_stats['successful_generations']/model_stats['total_requests']*100:.1f}%")
    with col3:
        st.metric("‚è±Ô∏è Avg Response", f"{model_stats['avg_response_time']:.1f}s")
    with col4:
        st.metric("Ìº°Ô∏è Avg Temperature", f"{model_stats['avg_temperature']:.2f}")
    with col5:
        st.metric("Ìæµ Top Genre", model_stats['most_popular_genre'])
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Ì≥à Usage Over Time")
        
        # Generate sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        usage_data = pd.DataFrame({
            'Date': dates,
            'Requests': np.random.poisson(40, 30) + np.sin(np.arange(30) * 0.2) * 10 + 40
        })
        
        fig = px.line(usage_data, x='Date', y='Requests', title="Daily API Requests")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Ìæµ Genre Distribution")
        
        genre_data = {
            'Electronic': 28,
            'Pop': 22,
            'Rock': 18,
            'Hip-Hop': 15,
            'Jazz': 8,
            'Classical': 6,
            'Other': 3
        }
        
        fig = px.pie(
            values=list(genre_data.values()),
            names=list(genre_data.keys()),
            title="Generated Playlists by Genre"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Embedding similarity analysis
    st.markdown("### Ì∑† Embedding Similarity Patterns")
    
    if 'embeddings' in st.session_state and len(st.session_state.embeddings) > 1:
        embeddings_data = st.session_state.embeddings
        
        # Calculate similarity matrix
        similarity_matrix = []
        labels = []
        
        for i, emb1 in enumerate(embeddings_data):
            row = []
            if i == 0:  # First iteration, set up labels
                for j, emb2 in enumerate(embeddings_data):
                    labels.append(f"Query {j+1}")
            
            for emb2 in embeddings_data:
                sim = cosine_similarity(emb1['embedding'], emb2['embedding'])
                row.append(sim)
            similarity_matrix.append(row)
        
        if similarity_matrix:
            fig = px.imshow(
                similarity_matrix,
                labels=dict(x="Query", y="Query", color="Similarity"),
                x=labels,
                y=labels,
                title="Embedding Similarity Matrix",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ì≤° Generate some embeddings to see similarity analysis!")
    
    # Download analytics
    st.markdown("### Ì≤æ Export Analytics")
    
    analytics_data = {
        'model_performance': model_stats,
        'usage_data': usage_data.to_dict('records') if 'usage_data' in locals() else [],
        'genre_distribution': genre_data,
        'generated_at': datetime.now().isoformat()
    }
    
    st.download_button(
        label="Ì≥• Download Analytics Report",
        data=json.dumps(analytics_data, indent=2),
        file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
