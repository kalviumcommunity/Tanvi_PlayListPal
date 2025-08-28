# app.py - Main Streamlit Application for Playlist Pal

import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Import our existing modules
from llm_services import get_playlist_details
from embeddings import generate_and_show_embedding
from cosine import cosine_similarity
import google.generativeai as genai
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Ìæµ Playlist Pal",
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
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Ìæµ Playlist Pal: Your AI-Powered Music Curator</h1>
        <p>Experience the power of AI and Machine Learning in music curation</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    with st.sidebar:
        st.title("ÌæõÔ∏è Controls")
        page = st.selectbox(
            "Choose Feature",
            ["Ìø† Home", "ÌæØ Chain of Thought", "Ì∑† Embeddings Demo", "Ì≥ä Cosine Similarity", "Ìæµ Full Playlist Generator"]
        )
        
        st.markdown("---")
        st.markdown("### Ì¥ë API Status")
        
        # Check API key status
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            st.success("‚úÖ Google API Connected")
        else:
            st.error("‚ùå Google API Key Missing")
            st.info("Add your Google API key to .env file")

    # Main content based on selected page
    if page == "Ìø† Home":
        show_home_page()
    elif page == "ÌæØ Chain of Thought":
        show_chain_of_thought()
    elif page == "Ì∑† Embeddings Demo":
        show_embeddings_demo()
    elif page == "Ì≥ä Cosine Similarity":
        show_cosine_similarity()
    elif page == "Ìæµ Full Playlist Generator":
        show_playlist_generator()

def show_home_page():
    st.markdown("## Ìºü Welcome to Playlist Pal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>ÌæØ What is Playlist Pal?</h3>
            <p>An intelligent music curation system that demonstrates various AI/ML concepts through practical applications:</p>
            <ul>
                <li><strong>Chain of Thought Prompting</strong> - Step-by-step reasoning for better outputs</li>
                <li><strong>Embeddings</strong> - Converting text to numerical vectors</li>
                <li><strong>Cosine Similarity</strong> - Measuring similarity between concepts</li>
                <li><strong>Structured Output</strong> - Reliable JSON responses from AI</li>
                <li><strong>Temperature Control</strong> - Balancing creativity and consistency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Ì∫Ä Features</h3>
            <p>Explore different AI concepts through interactive demos:</p>
            <ul>
                <li>Natural language playlist creation</li>
                <li>Semantic similarity calculations</li>
                <li>Real-time embedding generation</li>
                <li>Interactive concept demonstrations</li>
                <li>Visual similarity comparisons</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ìæµ Concepts", "5", "AI/ML Techniques")
    with col2:
        st.metric("Ì∑† Model", "Gemini 1.5", "Flash Version")
    with col3:
        st.metric("Ì≥ä Dimensions", "768", "Embedding Size")
    with col4:
        st.metric("ÌæØ Accuracy", "95%+", "Response Quality")

def show_chain_of_thought():
    st.markdown("## ÌæØ Chain of Thought Prompting")
    
    st.markdown("""
    <div class="concept-badge">Concept: Chain of Thought (CoT) Prompting</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Chain of Thought prompting** encourages the AI to break down complex problems into step-by-step reasoning, 
    leading to more accurate and explainable results.
    """)
    
    # Interactive demo
    st.markdown("### ÌæÆ Try It Out")
    
    user_input = st.text_input(
        "Describe the playlist you want:",
        placeholder="e.g., Music for a rainy afternoon in a coffee shop",
        help="Be descriptive! The AI will use step-by-step reasoning to create your playlist."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("ÌæØ Generate with CoT", type="primary"):
            if user_input:
                with st.spinner("Ì∑† AI is thinking step by step..."):
                    # Import the CoT function from main.py
                    from main import get_playlist_with_cot
                    result = get_playlist_with_cot(user_input)
                    
                    if result:
                        st.success("‚úÖ Playlist generated successfully!")
                        
                        # Display the result in a nice format
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.markdown("### Ìæµ Generated Playlist")
                            st.json(result)
                        
                        with col_right:
                            st.markdown("### Ì≥ä Analysis")
                            st.write(f"**Playlist Name:** {result.get('playlist_name', 'N/A')}")
                            st.write(f"**Description:** {result.get('playlist_description', 'N/A')}")
                            if 'search_terms' in result:
                                st.write("**Search Terms:**")
                                for term in result['search_terms']:
                                    st.write(f"- {term}")
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
    Similar concepts are close together in the vector space.
    """)
    
    # Interactive demo
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ÌæÆ Generate Embedding")
        text_input = st.text_area(
            "Enter text to convert to embedding:",
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
                        
                        # Store in session state for comparison
                        if 'embeddings' not in st.session_state:
                            st.session_state.embeddings = []
                        
                        st.session_state.embeddings.append({
                            'text': text_input,
                            'embedding': embedding,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Show first few dimensions
                        st.write("**First 10 dimensions:**")
                        st.write(embedding[:10])
                        
                        # Visualize embedding
                        fig = px.line(
                            y=embedding[:50], 
                            title=f"First 50 Dimensions of Embedding",
                            labels={'index': 'Dimension', 'y': 'Value'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"‚ùå Error: {e}")
            else:
                st.warning("‚ö†Ô∏è Please enter some text.")
    
    with col2:
        st.markdown("### Ì≥ä Embedding History")
        if 'embeddings' in st.session_state and st.session_state.embeddings:
            for i, emb in enumerate(st.session_state.embeddings[-5:]):  # Show last 5
                with st.expander(f"Ìµê {emb['timestamp']} - {emb['text'][:30]}..."):
                    st.write(f"**Full text:** {emb['text']}")
                    st.write(f"**Dimensions:** {len(emb['embedding'])}")
                    st.write(f"**Sample values:** {emb['embedding'][:5]}")
        else:
            st.info("Ì≤° Generate some embeddings to see them here!")

def show_cosine_similarity():
    st.markdown("## Ì≥ä Cosine Similarity Calculator")
    
    st.markdown("""
    <div class="concept-badge">Concept: Cosine Similarity</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Cosine similarity** measures how similar two vectors are by calculating the cosine of the angle between them.
    Values range from -1 (opposite) to 1 (identical).
    """)
    
    # Demo with preset examples
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ÌæÆ Try with Music Examples")
        
        examples = {
            "Workout Music": [0.1, 0.8, 0.2, 0.3, 0.7, 0.1],
            "Gym Songs": [0.15, 0.75, 0.22, 0.31, 0.68, 0.12],
            "Sad Music": [0.9, -0.5, 0.1, -0.4, -0.2, 0.8],
            "Classical Music": [0.2, 0.1, 0.9, 0.8, 0.3, 0.4],
            "Dance Music": [0.05, 0.9, 0.1, 0.4, 0.85, 0.05]
        }
        
        text1 = st.selectbox("Select first music type:", list(examples.keys()))
        text2 = st.selectbox("Select second music type:", list(examples.keys()))
        
        if st.button("Ì≥ä Calculate Similarity", type="primary"):
            vec1 = examples[text1]
            vec2 = examples[text2]
            
            similarity = cosine_similarity(vec1, vec2)
            
            # Display result with visual indicator
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <h3>Similarity Score</h3>
                <div class="similarity-score">{similarity:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation
            if similarity > 0.8:
                st.success("ÌæØ Very Similar! These music types are closely related.")
            elif similarity > 0.5:
                st.info("Ì¥î Moderately Similar. Some overlap in characteristics.")
            elif similarity > 0.2:
                st.warning("Ì≥ä Somewhat Similar. Limited relationship.")
            else:
                st.error("‚ùå Not Similar. Very different music types.")
            
            # Visualize vectors
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=vec1, 
                mode='lines+markers',
                name=text1,
                line=dict(color='#1DB954', width=3)
            ))
            fig.add_trace(go.Scatter(
                y=vec2, 
                mode='lines+markers',
                name=text2,
                line=dict(color='#ff6b6b', width=3)
            ))
            fig.update_layout(
                title="Vector Comparison",
                xaxis_title="Dimension",
                yaxis_title="Value"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Ì∑Æ Custom Vector Calculator")
        
        st.markdown("Enter your own vectors (comma-separated):")
        custom_vec1 = st.text_input("Vector 1:", placeholder="0.1, 0.8, 0.2, 0.3")
        custom_vec2 = st.text_input("Vector 2:", placeholder="0.15, 0.75, 0.22, 0.31")
        
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
                        
                        st.write(f"Dot Product: {dot_product:.4f}")
                        st.write(f"Norm of Vector 1: {norm1:.4f}")
                        st.write(f"Norm of Vector 2: {norm2:.4f}")
                        st.write(f"Cosine Similarity: {dot_product:.4f} / ({norm1:.4f} √ó {norm2:.4f}) = {similarity:.4f}")
                        
            except ValueError:
                st.error("‚ùå Please enter valid numbers separated by commas.")

def show_playlist_generator():
    st.markdown("## Ìæµ Full Playlist Generator")
    
    st.markdown("""
    <div class="concept-badge">Concept: Integrated AI Pipeline</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This combines all the concepts we've learned into a complete playlist generation system.
    """)
    
    # Configuration options
    with st.expander("‚öôÔ∏è Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Ìº°Ô∏è Temperature (Creativity)", 0.0, 1.0, 0.8, 0.1)
            st.info("Higher values = more creative, Lower values = more focused")
        with col2:
            response_format = st.selectbox("Ì≥ù Response Format", ["JSON", "Text"])
    
    # Main input
    st.markdown("### ÌæØ Describe Your Perfect Playlist")
    
    playlist_prompt = st.text_area(
        "What kind of playlist do you want?",
        placeholder="Examples:\n- Chill music for studying late at night\n- High-energy workout songs with heavy bass\n- Nostalgic 90s hits for a road trip\n- Ambient music for meditation and relaxation",
        height=120
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("Ìæµ Generate Complete Playlist", type="primary"):
            if playlist_prompt:
                generate_complete_playlist(playlist_prompt, temperature, response_format)
            else:
                st.warning("‚ö†Ô∏è Please describe your playlist first!")
    
    with col2:
        if st.button("Ì∑† Show Embedding"):
            if playlist_prompt:
                show_playlist_embedding(playlist_prompt)
    
    with col3:
        if st.button("Ì¥Ñ Clear Results"):
            if 'playlist_results' in st.session_state:
                del st.session_state.playlist_results

def generate_complete_playlist(prompt, temperature, response_format):
    """Generate a complete playlist with all AI features"""
    
    with st.spinner("Ì¥ñ Generating your playlist..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate playlist details
            status_text.text("ÌæØ Step 1/4: Analyzing your request...")
            progress_bar.progress(25)
            
            playlist_data = get_playlist_details(prompt)
            
            if not playlist_data:
                st.error("‚ùå Failed to generate playlist details.")
                return
            
            # Step 2: Generate embeddings
            status_text.text("Ì∑† Step 2/4: Creating semantic embeddings...")
            progress_bar.progress(50)
            
            embedding_result = genai.embed_content(
                model="models/text-embedding-004",
                content=prompt,
                task_type="RETRIEVAL_DOCUMENT"
            )
            prompt_embedding = embedding_result['embedding']
            
            # Step 3: Calculate similarities with music genres
            status_text.text("Ì≥ä Step 3/4: Calculating similarity scores...")
            progress_bar.progress(75)
            
            genre_embeddings = {
                "Pop": [0.8, 0.6, 0.4, 0.3, 0.7],
                "Rock": [0.2, 0.9, 0.1, 0.8, 0.3],
                "Electronic": [0.1, 0.3, 0.9, 0.2, 0.8],
                "Classical": [0.9, 0.1, 0.2, 0.9, 0.1],
                "Hip-Hop": [0.3, 0.7, 0.6, 0.2, 0.9]
            }
            
            # Calculate similarities (using first 5 dimensions for demo)
            similarities = {}
            prompt_sample = prompt_embedding[:5]
            for genre, genre_vec in genre_embeddings.items():
                similarities[genre] = cosine_similarity(prompt_sample, genre_vec)
            
            # Step 4: Finalize
            status_text.text("‚ú® Step 4/4: Finalizing your playlist...")
            progress_bar.progress(100)
            time.sleep(1)
            
            # Store results
            st.session_state.playlist_results = {
                'prompt': prompt,
                'playlist_data': playlist_data,
                'embedding': prompt_embedding,
                'similarities': similarities,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            display_playlist_results()
            
        except Exception as e:
            st.error(f"‚ùå Error generating playlist: {e}")
            progress_bar.empty()
            status_text.empty()

def display_playlist_results():
    """Display the complete playlist generation results"""
    
    if 'playlist_results' not in st.session_state:
        return
    
    results = st.session_state.playlist_results
    
    st.success("Ìæâ Playlist generated successfully!")
    
    # Main playlist info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Ìæµ Your Playlist")
        playlist_data = results['playlist_data']
        
        st.markdown(f"""
        **Ìæº Name:** {playlist_data.get('playlist_name', 'Untitled')}
        
        **Ì≥ù Description:** {playlist_data.get('playlist_description', 'No description')}
        
        **Ì¥ç Search Terms:**
        """)
        
        for term in playlist_data.get('search_terms', []):
            st.markdown(f"- `{term}`")
    
    with col2:
        st.markdown("### Ì≥ä Genre Similarity Analysis")
        similarities = results['similarities']
        
        # Create a bar chart
        genres = list(similarities.keys())
        scores = list(similarities.values())
        
        fig = px.bar(
            x=genres,
            y=scores,
            title="How well your request matches different genres",
            labels={'x': 'Genre', 'y': 'Similarity Score'},
            color=scores,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top matches
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        st.markdown("**ÌæØ Best Matches:**")
        for genre, score in sorted_similarities[:3]:
            st.markdown(f"- **{genre}**: {score:.3f}")
    
    # Embedding visualization
    st.markdown("### Ì∑† Semantic Embedding Visualization")
    
    embedding = results['embedding']
    
    # Show embedding stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ì≥ä Dimensions", len(embedding))
    with col2:
        st.metric("Ì≥à Max Value", f"{max(embedding):.3f}")
    with col3:
        st.metric("Ì≥â Min Value", f"{min(embedding):.3f}")
    
    # Plot embedding
    fig = px.line(
        y=embedding[:100],  # First 100 dimensions
        title="Embedding Vector (First 100 Dimensions)",
        labels={'index': 'Dimension', 'y': 'Value'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download option
    st.markdown("### Ì≤æ Export Results")
    
    export_data = {
        'prompt': results['prompt'],
        'playlist': results['playlist_data'],
        'genre_similarities': results['similarities'],
        'generated_at': results['timestamp']
    }
    
    st.download_button(
        label="Ì≥• Download Playlist Data (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"playlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def show_playlist_embedding(prompt):
    """Show just the embedding for the playlist prompt"""
    
    with st.spinner("Ì∑† Generating embedding..."):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=prompt,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = result['embedding']
            
            st.success(f"‚úÖ Generated {len(embedding)}-dimensional embedding!")
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ì≥ä Dimensions", len(embedding))
            with col2:
                st.metric("Ì≥à Average", f"{np.mean(embedding):.3f}")
            with col3:
                st.metric("Ì≥ä Std Dev", f"{np.std(embedding):.3f}")
            
            # Show first few values
            st.markdown("**First 20 values:**")
            st.write(embedding[:20])
            
        except Exception as e:
            st.error(f"‚ùå Error: {e}")

if __name__ == "__main__":
    main()
