"""
analytics_engine.py - Advanced Analytics Engine for Playlist Pal

This module provides comprehensive analytics capabilities including:
- User behavior analysis
- AI model performance metrics
- Music preference insights
- Playlist similarity clustering
- Real-time usage monitoring
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

class PlaylistAnalytics:
    """Main analytics engine for Playlist Pal"""
    
    def __init__(self, db_path: str = "playlist_analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for analytics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for analytics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                ip_address TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS playlist_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                prompt TEXT,
                playlist_data TEXT,
                generation_time REAL,
                temperature REAL,
                success BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                input_text TEXT,
                embedding_data TEXT,
                calculation_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS similarity_calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                vector1_data TEXT,
                vector2_data TEXT,
                similarity_score REAL,
                calculation_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spotify_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                action_type TEXT,
                spotify_data TEXT,
                success BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_user_session(self, session_id: str, user_agent: str = None, ip_address: str = None):
        """Log a new user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO user_sessions (session_id, user_agent, ip_address)
            VALUES (?, ?, ?)
        """, (session_id, user_agent, ip_address))
        
        conn.commit()
        conn.close()
    
    def log_playlist_generation(self, session_id: str, prompt: str, playlist_data: Dict, 
                              generation_time: float, temperature: float, success: bool):
        """Log a playlist generation event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO playlist_generations 
            (session_id, prompt, playlist_data, generation_time, temperature, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, prompt, json.dumps(playlist_data), generation_time, temperature, success))
        
        conn.commit()
        conn.close()
    
    def get_usage_statistics(self, days: int = 30) -> Dict:
        """Get comprehensive usage statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Total sessions
        total_sessions = pd.read_sql_query("""
            SELECT COUNT(DISTINCT session_id) as count
            FROM user_sessions
            WHERE timestamp >= ?
        """, conn, params=(start_date,))['count'].iloc[0]
        
        # Total playlist generations
        total_generations = pd.read_sql_query("""
            SELECT COUNT(*) as count
            FROM playlist_generations
            WHERE timestamp >= ?
        """, conn, params=(start_date,))['count'].iloc[0]
        
        # Success rate
        success_data = pd.read_sql_query("""
            SELECT success, COUNT(*) as count
            FROM playlist_generations
            WHERE timestamp >= ?
            GROUP BY success
        """, conn, params=(start_date,))
        
        success_rate = 0
        if not success_data.empty:
            total_attempts = success_data['count'].sum()
            successful_attempts = success_data[success_data['success'] == 1]['count'].sum() if any(success_data['success'] == 1) else 0
            success_rate = (successful_attempts / total_attempts) * 100 if total_attempts > 0 else 0
        
        # Average generation time
        avg_gen_time = pd.read_sql_query("""
            SELECT AVG(generation_time) as avg_time
            FROM playlist_generations
            WHERE timestamp >= ? AND success = 1
        """, conn, params=(start_date,))['avg_time'].iloc[0] or 0
        
        conn.close()
        
        return {
            'total_sessions': int(total_sessions),
            'total_generations': int(total_generations),
            'success_rate': float(success_rate),
            'avg_generation_time': float(avg_gen_time),
            'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }
    
    def generate_insights_report(self) -> Dict:
        """Generate comprehensive insights report"""
        usage_stats = self.get_usage_statistics()
        
        # Generate insights
        insights = []
        
        # Usage insights
        if usage_stats['success_rate'] > 90:
            insights.append("ÔøΩÔøΩ Excellent AI performance with >90% success rate!")
        elif usage_stats['success_rate'] > 70:
            insights.append("Ì±ç Good AI performance, room for improvement.")
        else:
            insights.append("‚ö†Ô∏è AI performance needs attention.")
        
        # Performance insights
        if usage_stats['avg_generation_time'] < 3:
            insights.append("‚ö° Fast response times - great user experience!")
        elif usage_stats['avg_generation_time'] < 5:
            insights.append("‚è±Ô∏è Acceptable response times.")
        else:
            insights.append("Ì∞å Slow response times may affect user experience.")
        
        return {
            'summary_stats': usage_stats,
            'insights': insights,
            'generated_at': datetime.now().isoformat()
        }

def create_analytics_dashboard():
    """Create Streamlit analytics dashboard"""
    
    st.title("Ì≥ä Playlist Pal Analytics Dashboard")
    
    # Initialize analytics
    if 'analytics' not in st.session_state:
        st.session_state.analytics = PlaylistAnalytics()
    
    analytics = st.session_state.analytics
    
    # Sidebar controls
    with st.sidebar:
        st.header("Ì≥ä Analytics Controls")
        
        days_range = st.slider("Days to analyze", 1, 90, 30)
        refresh_data = st.button("Ì¥Ñ Refresh Data")
        
        if refresh_data:
            st.rerun()
    
    # Main dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ì≥à Usage Overview")
        
        # Get usage statistics
        usage_stats = analytics.get_usage_statistics(days_range)
        
        # Display metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Ì±• Total Sessions", usage_stats['total_sessions'])
        with metric_cols[1]:
            st.metric("Ìæµ Generations", usage_stats['total_generations'])
        with metric_cols[2]:
            st.metric("‚úÖ Success Rate", f"{usage_stats['success_rate']:.1f}%")
        with metric_cols[3]:
            st.metric("‚è±Ô∏è Avg Time", f"{usage_stats['avg_generation_time']:.1f}s")
    
    with col2:
        st.header("ÌæØ Quick Insights")
        
        insights_report = analytics.generate_insights_report()
        
        for insight in insights_report['insights']:
            st.write(insight)
    
    # Export options
    st.markdown("### Ì≤æ Export Analytics Data")
    
    full_report = analytics.generate_insights_report()
    
    st.download_button(
        label="Ì≥• Download Full Report (JSON)",
        data=json.dumps(full_report, indent=2),
        file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    create_analytics_dashboard()
