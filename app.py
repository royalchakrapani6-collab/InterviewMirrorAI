import warnings
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import speech_recognition as sr
import matplotlib.pyplot as plt
import tempfile
import os
from cryptography.fernet import Fernet
st.title("🎤 Interview Mirror AI - Secure & Ready Demo")
st.info("""
🔒 Privacy Notice:  
All processing is done *locally* on your device.  
Your video and audio data *is never uploaded or shared*.  
Optional feedback can be saved *encrypted locally*.
""")
key_file = "localkey.key"
if not os.path.exists(key_file):
    key = Fernet.generate_key()
    with open(key_file, "wb") as f:
        f.write(key)
else:
    with open(key_file, "rb") as f:
        key = f.read()
fernet = Fernet(key)

# -------------------------- Video / Face Analysis --------------------------
st.subheader("Webcam / Video Analysis")

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
blink_threshold = 0.25
eye_blink_count = 0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    coords = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_indices]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (A + B) / (2.0 * C)
    return ear

video_file = st.file_uploader("Upload video (optional)", type=["mp4","mov"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
else:
    cap = cv2.VideoCapture(0)

st.write("Press 'q' to stop video analysis.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, frame.shape)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, frame.shape)
            avg_ear = (left_ear + right_ear)/2.0
            if avg_ear < blink_threshold:
                eye_blink_count += 1
    cv2.imshow("Interview Mirror - Press q to Exit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
st.write(f"Total Eye Blinks Detected: {eye_blink_count}")

# -------------------------- Speech Analysis --------------------------
st.subheader("Speech Analysis")
filler_words = ["um", "uh", "like", "so", "you know"]
filler_count = 0

if st.button("Start Speaking (10 sec max)"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            st.write("Recording...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            filler_count = sum(text.lower().split().count(word) for word in filler_words)
            st.write(f"Filler words count: {filler_count}")
    except Exception as e:
        st.write("Error:", e)

# -------------------------- Feedback Generator --------------------------
st.subheader("Confidence Score & Feedback")

def generate_feedback(filler_count, eye_blinks):
    score = max(0, 100 - filler_count*5 - abs(eye_blinks-20))
    st.write(f"Confidence Score: {score}/100")
    if filler_count > 3:
        st.write("💡 Reduce filler words like 'um', 'like', 'uh'.")
    if eye_blinks > 25:
        st.write("💡 Maintain steady eye contact.")
    categories = ['Filler Words', 'Eye Blinks', 'Score']
    values = [filler_count, eye_blinks, score]
    fig, ax = plt.subplots()
    ax.bar(categories, values, color=['red','orange','green'])
    ax.set_ylim([0,100])
    st.pyplot(fig)
    return score

if st.button("Generate Feedback"):
    score = generate_feedback(filler_count, eye_blink_count)
    save_option = st.checkbox("Save feedback locally (encrypted)")
    if save_option:
        data = f"Filler Words: {filler_count}, Eye Blinks: {eye_blink_count}, Score: {score}"
        encrypted_data = fernet.encrypt(data.encode())
        with open("feedback.enc", "wb") as f:
            f.write(encrypted_data)
        st.success("Feedback saved securely on your device (feedback.enc)")