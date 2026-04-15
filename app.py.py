import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import pandas as pd
import plotly.graph_objects as go

# Load trained model
model = load_model("deepfake_model.h5")

st.set_page_config(page_title="Deepfake Detection", layout="centered", page_icon="🔍")

# Custom CSS for black and yellow theme
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }
    
    /* Title styling */
    h1 {
        color: #FFD700 !important;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(255, 215, 0, 0.3);
        padding: 20px 0;
        border-bottom: 3px solid #FFD700;
        margin-bottom: 30px;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #FFD700 !important;
        font-family: 'Arial', sans-serif;
        margin-top: 20px;
        border-left: 4px solid #FFD700;
        padding-left: 10px;
    }
    
    /* Text styling */
    p, .stMarkdown {
        color: #E0E0E0 !important;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background-color: #1a1a1a;
        border: 2px dashed #FFD700;
        border-radius: 10px;
        padding: 20px;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #FFA500;
        background-color: #2a2a2a;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000000;
        font-weight: bold;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 215, 0, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FFA500, #FFD700);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 215, 0, 0.5);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #FFD700;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(0, 255, 0, 0.1);
        border: 2px solid #00FF00;
        border-radius: 10px;
        padding: 15px;
        color: #00FF00 !important;
    }
    
    .stError {
        background-color: rgba(255, 0, 0, 0.1);
        border: 2px solid #FF0000;
        border-radius: 10px;
        padding: 15px;
        color: #FF0000 !important;
    }
    
    /* Video container */
    .stVideo {
        border: 3px solid #FFD700;
        border-radius: 10px;
        padding: 5px;
        background-color: #1a1a1a;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(255, 215, 0, 0.2);
    }
    
    [data-testid="metric-container"] label {
        color: #FFD700 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    /* Image container */
    .stImage {
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 5px;
        background-color: #1a1a1a;
    }
    
    /* Caption text */
    .stImage > div > div > div > div {
        color: #FFD700 !important;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header with icon
st.markdown("<h1>🔍 Deepfake Video Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FFD700; font-size: 18px;'>Advanced AI-Powered Authentication</p>", unsafe_allow_html=True)

# Info box
st.markdown("""
<div style='background-color: #1a1a1a; border: 2px solid #FFD700; border-radius: 10px; padding: 20px; margin: 20px 0;'>
    <p style='color: #FFD700; font-weight: bold; margin: 0;'>📋 Instructions:</p>
    <ul style='color: #E0E0E0; margin-top: 10px;'>
        <li>Upload a video file (MP4, AVI, or MOV format)</li>
        <li>Click "Detect Deepfake" to analyze</li>
        <li>View detailed frame-by-frame analysis</li>
    </ul>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if uploaded_file is not None:
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Display video with custom styling
    st.video(uploaded_file)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        detect_button = st.button("🔍 Detect Deepfake", use_container_width=True)

    if detect_button:
        
        cap = cv2.VideoCapture(tfile.name)
        
        frame_count = 0
        fake_scores = []
        frame_numbers = []
        frames_list = []
        
        st.markdown("<p style='color: #FFD700; font-weight: bold;'>⏳ Analyzing video frames...</p>", unsafe_allow_html=True)
        progress = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 30 == 0:
                resized = cv2.resize(frame,(224,224))
                normalized = resized / 255.0
                input_frame = np.expand_dims(normalized,axis=0)
                
                prediction = model.predict(input_frame)
                fake_prob = prediction[0][0]
                
                fake_scores.append(fake_prob)
                frame_numbers.append(frame_count)
                frames_list.append(frame)
            
            frame_count += 1
            progress.progress(min(frame_count/500,1.0))
        
        cap.release()
        
        avg_fake = np.mean(fake_scores)
        real_prob = 1 - avg_fake
        
        # Results section
        st.markdown("<h2>📊 Detection Results</h2>", unsafe_allow_html=True)
        
        # Result box with conditional styling
        if avg_fake > 0.5:
            st.markdown(f"""
            <div style='background-color: rgba(255, 0, 0, 0.1); border: 3px solid #FF0000; border-radius: 15px; padding: 20px; text-align: center;'>
                <h3 style='color: #FF0000; margin: 0;'>⚠️ FAKE VIDEO DETECTED</h3>
                <p style='color: #FF6B6B; font-size: 18px; margin-top: 10px;'>This video appears to be manipulated</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: rgba(0, 255, 0, 0.1); border: 3px solid #00FF00; border-radius: 15px; padding: 20px; text-align: center;'>
                <h3 style='color: #00FF00; margin: 0;'>✅ AUTHENTIC VIDEO</h3>
                <p style='color: #90EE90; font-size: 18px; margin-top: 10px;'>This video appears to be genuine</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability metrics
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🚫 Fake Probability",
                value=f"{avg_fake*100:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label="✅ Real Probability",
                value=f"{real_prob*100:.2f}%",
                delta=None
            )
        
        # Most suspicious frame
        st.markdown("<h2>🎯 Most Suspicious Frame</h2>", unsafe_allow_html=True)
        
        max_index = np.argmax(fake_scores)
        most_fake_frame = frames_list[max_index]
        most_fake_frame_number = frame_numbers[max_index]
        most_fake_prob = fake_scores[max_index]
        
        # Convert BGR to RGB for display
        most_fake_frame_rgb = cv2.cvtColor(most_fake_frame, cv2.COLOR_BGR2RGB)
        
        st.image(
            most_fake_frame_rgb,
            caption=f"Frame {most_fake_frame_number} | Fake Probability: {most_fake_prob*100:.2f}%",
            use_column_width=True
        )
        
        # Frame analysis chart with Plotly
        st.markdown("<h2>📈 Frame-by-Frame Analysis</h2>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frame_numbers,
            y=[score * 100 for score in fake_scores],
            mode='lines+markers',
            name='Fake Probability',
            line=dict(color='#FFD700', width=3),
            marker=dict(color='#FFA500', size=8),
            fill='tozeroy',
            fillcolor='rgba(255, 215, 0, 0.2)'
        ))
        
        # Add threshold line
        fig.add_hline(y=50, line_dash="dash", line_color="#FF0000", 
                     annotation_text="Detection Threshold (50%)")
        
        fig.update_layout(
            title="Deepfake Probability Across Frames",
            xaxis_title="Frame Number",
            yaxis_title="Fake Probability (%)",
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#FFD700'),
            height=400,
            hovermode='x unified',
            showlegend=False
        )
        
        fig.update_xaxes(gridcolor='#333333', zeroline=False)
        fig.update_yaxes(gridcolor='#333333', zeroline=False, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("<h2>📊 Analysis Summary</h2>", unsafe_allow_html=True)
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Total Frames Analyzed", len(frame_numbers))
        
        with summary_col2:
            st.metric("Max Fake Probability", f"{max(fake_scores)*100:.2f}%")
        
        with summary_col3:
            st.metric("Min Fake Probability", f"{min(fake_scores)*100:.2f}%")

else:
    st.markdown("""
    <div style='background-color: #1a1a1a; border: 2px solid #FFD700; border-radius: 10px; padding: 30px; text-align: center; margin-top: 50px;'>
        <p style='color: #FFD700; font-size: 20px; margin: 0;'>👆 Upload a video to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)
