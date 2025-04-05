import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf
import gdown
import os

# Model loader
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

# Frame preprocessing
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype(np.float32) / 127.5 - 1
    return np.expand_dims(frame, axis=0)

# Postprocessing
def postprocess_frame(output):
    frame = (output[0] + 1) * 127.5
    return np.clip(frame, 0, 255).astype(np.uint8)

# WebRTC VideoProcessor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.ready = False

    def update_model(self, model):
        self.model = model
        self.ready = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.ready and self.model:
            input_frame = preprocess_frame(img)
            output = self.model.predict(input_frame)
            output_frame = postprocess_frame(output)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            return av.VideoFrame.from_ndarray(output_frame, format="bgr24")
        return frame

# UI
st.title("📷 Real-Time Dehazing (WebRTC-enabled)")

# Select mode
mode = st.radio("Choose input method:", ["Webcam", "Upload Image", "Upload Video"])

# Select location
location = st.selectbox("Select location:", ["Patiala", "Thapar Campus"])

# Load model once based on location
model = load_model(location)

# Store model in session state
# Ensure model and processor are initialized
# Only create processor if not already in session state
if "processor" not in st.session_state:
    class VideoProcessor:
        def __init__(self):
            self.model = None

        def update_model(self, model):
            self.model = model

        def recv(self, frame):
            import av
            import cv2
            import numpy as np
            img = frame.to_ndarray(format="bgr24")
            img_resized = cv2.resize(img, (256, 256))
            img_normalized = (img_resized / 127.5) - 1.0
            img_input = np.expand_dims(img_normalized, axis=0)

            output = self.model.predict(img_input)[0]
            output = ((output + 1) * 127.5).astype(np.uint8)
            output = cv2.resize(output, (img.shape[1], img.shape[0]))
            return av.VideoFrame.from_ndarray(output, format="bgr24")

    st.session_state.processor = VideoProcessor()

# Update model every time user selects location
st.session_state.processor.update_model(model)

# Safe to call webrtc_streamer now
if mode == "Webcam":
    st.info("Make sure to allow webcam access.")
    webrtc_streamer(
        key="dehazing",
        video_processor_factory=lambda: st.session_state.processor,
        media_stream_constraints={"video": True, "audio": False}
    )
ream_constraints={"video": True, "audio": False}
)

elif mode == "Upload Image":
    uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        img = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), 1)
        input_img = preprocess_frame(img)
        output_img = postprocess_frame(model.predict(input_img))
        st.image([img, output_img], caption=["Original", "Dehazed"], channels="BGR")

elif mode == "Upload Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        tfile = f"temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_vid.read())
        st.video(tfile)
        st.warning("Video dehazing not yet implemented frame-by-frame in cloud. Try image or webcam.")

