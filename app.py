import streamlit as st
import cv2
import time
from main1 import initialize_detection, process_frame

# Streamlit setup
st.title("Driver Drowsiness Detection")

st.write("Use your webcam for real-time detection.")

# Settings
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
MOUTH_AR_THRESH = 0.79
shape_predictor = "shape_predictor_68_face_landmarks.dat"

detector, predictor = initialize_detection(shape_predictor)

start_detection = st.button("Start Detection")
stop_detection = st.button("Stop Detection")

if start_detection:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    st.session_state['detection_running'] = True

    COUNTER = 0
    while cap.isOpened() and st.session_state.get('detection_running', False):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, COUNTER, ear, mar = process_frame(frame, detector, predictor, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, COUNTER, MOUTH_AR_THRESH)
        
        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        stframe.image(frame, channels="RGB")
        
        # Display EAR and MAR
        st.write(f"EAR: {ear:.2f}, MAR: {mar:.2f}")
        
        # Pause to ensure frame rate is not too high
        time.sleep(0.03)

    cap.release()

if stop_detection:
    st.session_state['detection_running'] = False
    st.warning("Detection stopped and webcam turned off.")
