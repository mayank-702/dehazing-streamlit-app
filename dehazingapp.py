import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf
import gdown
import os

# ----------------------
# Load the ML Model
# ----------------------
@st.cache_resource
def load_model(location):
    if location == "Patiala":
        file_id = "19hbgk6afotZ6rt_LV9y8qK7Jiw3kWRbQ"
        filename = "mixed_noaug.keras"
    else:
        file_id = "1HIwQqPoZShblcuG4Sc2kJ5qoi4w18ZTS"
        filename = "pix2pix.keras"

    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

    return tf.keras.models.load_model(filename)


# ----------------------
# Frame Pre/Postprocessing
# ----------------------
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype(np.float32) / 127.5 - 1
    return np.expand_dims(frame, axis=0)

def postprocess_frame(output):
    frame = (output[0] + 1) * 127.5
    return np.clip(frame, 0, 255).astype(np.uint8)


# ----------------------
# Video Processor Class
# ----------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.counter = 0  # To skip frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Optional: skip every other frame to reduce load
        self.counter += 1
        if self.counter % 2 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            input_img = preprocess_frame(img)
            output = self.model.predict(input_img)
            result = postprocess_frame(output)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            return av.VideoFrame.from_ndarray(result_bgr, format="bgr24")
        except Exception as e:
            print("Prediction error:", e)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------
# Streamlit UI
# ----------------------
st.title("📷 Real-Time Dehazing (WebRTC-enabled)")

mode = st.radio("Choose input method:", ["Webcam", "Upload Image", "Upload Video"])
location = st.selectbox("Select location:", ["Patiala", "Thapar Campus"])

model = load_model(location)

# ----------------------
# Webcam Stream
# ----------------------
if mode == "Webcam":
    st.info("Allow webcam access when prompted.")

    webrtc_streamer(
        key="dehazing",
        video_processor_factory=lambda: VideoProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ----------------------
# Image Upload
# ----------------------
elif mode == "Upload Image":
    uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        img = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), 1)
        input_img = preprocess_frame(img)
        output_img = postprocess_frame(model.predict(input_img))
        st.image([img, output_img], caption=["Original", "Dehazed"], channels="BGR")

# ----------------------
# Video Upload
# ----------------------
elif mode == "Upload Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        tfile = f"temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_vid.read())
        st.video(tfile)
        st.warning("Frame-by-frame dehazing for videos isn't implemented yet.")
