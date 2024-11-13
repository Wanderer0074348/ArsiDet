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


def main():
    st.title("Arabic Sign Language Detection")

    model = load_model()

    video_placeholder = st.empty()

    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    if not st.session_state.camera_on:
        if st.button("Start Camera", key="start_camera"):
            st.session_state.camera_on = True
    else:
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


if __name__ == "__main__":
    main()