import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# App title
st.set_page_config(page_title="Dehazing App")
st.title("Real-Time Dehazing App")

# Model selector
device_location = st.selectbox("Select your location:", ["Patiala", "Thapar Campus"])

# Load model based on selection
@st.cache_resource(show_spinner=False)
def load_model(location):
    if location == "Patiala":
        return tf.keras.models.load_model("mixed_noaug.keras")
    else:
        return tf.keras.models.load_model("pix2pix.keras")

model = load_model(device_location)

# Image preprocessing/postprocessing
def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = (frame / 127.5) - 1
    return np.expand_dims(frame, axis=0)

def postprocess_frame(frame_array):
    frame_array = (frame_array[0] + 1) * 127.5
    return np.clip(frame_array, 0, 255).astype(np.uint8)

# Webcam (local OpenCV)
if st.button("Open Webcam (Local Only)"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_webcam = st.button("Stop Webcam")

    while cap.isOpened() and not stop_webcam:
        ret, frame = cap.read()
        if not ret:
            break
        processed = preprocess_frame(frame)
        output = model.predict(processed)
        output_img = postprocess_frame(output)
        stframe.image(output_img, channels="RGB")
    cap.release()

# Webcam via browser (for cloud)
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = preprocess_frame(img)
        output = model.predict(processed)
        output_img = postprocess_frame(output)
        return output_img

if st.button("Open Webcam (Cloud)"):
    webrtc_streamer(key="cloud_cam", video_processor_factory=VideoProcessor)

# Upload video
uploaded_video = st.file_uploader("Upload a video for dehazing", type=["mp4", "avi"])
if uploaded_video is not None:
    st.video(uploaded_video)
    temp_file = f"temp_video.mp4"
    with open(temp_file, "wb") as f:
        f.write(uploaded_video.read())
    cap = cv2.VideoCapture(temp_file)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = preprocess_frame(frame)
        output = model.predict(processed)
        output_img = postprocess_frame(output)
        stframe.image(output_img, channels="RGB")
    cap.release()

# Upload image
uploaded_image = st.file_uploader("Upload an image for dehazing", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image.convert("RGB"))
    processed = preprocess_frame(image_np)
    output = model.predict(processed)
    output_img = postprocess_frame(output)
    st.image(output_img, caption="Dehazed Image", channels="RGB")
