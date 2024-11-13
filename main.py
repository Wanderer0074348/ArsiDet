import streamlit as st
from ultralytics import YOLO
import cv2
import torch
import numpy as np

@st.cache_resource
def load_model():
    model = YOLO('models/ArabicSignLanguage60.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

def set_custom_style():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo&family=Amiri&display=swap');

            .stApp {
                background-color: #f0f0f0;
                font-family: 'Cairo', sans-serif;
            }

            h1 {
                color: #16a085;
                font-family: 'Amiri', serif;
                font-size: 40px;
                font-weight: bold;
                text-align: center;
                margin-top: 30px;
                margin-bottom: 30px;
            }

            .stButton > button {
                background-color: #16a085;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 20px;
                font-family: 'Cairo', sans-serif;
            }

            .stButton > button:hover {
                background-color: #1abc9c;
            }

            .stImage > img {
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }

            .streamlit-expanderHeader {
                font-size: 18px;
                font-weight: bold;
                color: #16a085;
                font-family: 'Amiri', serif;
            }

            .stAlert > div {
                background-color: #f1c40f;
                color: #333;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Cairo', sans-serif;
            }
        </style>
    """, unsafe_allow_html=True)

def main():
    set_custom_style()

    st.title("Arabic Sign Language Detection")

    model = load_model()

    video_placeholder = st.empty()

    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.camera_on:
            if st.button("Start Camera", key="start_camera"):
                st.session_state.camera_on = True
    with col2:
        if st.session_state.camera_on:
            if st.button("Stop Camera", key="stop_camera"):
                st.session_state.camera_on = False

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0)

        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera")
                break

            results = model(frame, conf=0.25)

            result_frame = results[0].plot()

            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(
                result_frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

    st.markdown("""
    <div style="text-align: center; margin-top: 50px; font-family: 'Cairo', sans-serif;">
        <h2 style="color: #16a085;">About Arabic Sign Language Detection</h2>
        <p style="color:#000000;>This application uses advanced AI to detect and interpret Arabic sign language in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()