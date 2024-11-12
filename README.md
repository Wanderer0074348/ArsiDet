# Arabic Sign Language Detection

This project uses a YOLO model for Arabic Sign Language detection with a Streamlit interface.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/Wanderer0074348/ArsiDet.git
   cd ArsiDet
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure your webcam is connected and functioning.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Your default web browser should open automatically. If not, navigate to the URL shown in the terminal (usually `http://localhost:8501`).

4. In the Streamlit interface:
   - Click "Start Camera" to begin the sign language detection.
   - The video feed will appear with real-time detections.
   - Detection results will be displayed below the video feed.
   - Click "Stop Camera" to end the session.

## Troubleshooting

- If you encounter any issues with the camera, ensure that your system permissions allow access to the webcam.
- For CUDA-related errors, make sure you have the appropriate CUDA toolkit installed for your GPU.

## Requirements

The main requirements for this project are:
- Python 3.7+
- Streamlit
- Ultralytics YOLO
- OpenCV
- PyTorch

For a complete list of dependencies, refer to the `requirements.txt` file.

