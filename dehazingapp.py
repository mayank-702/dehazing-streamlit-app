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



import threading

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.ready = False
        self.output_frame = None
        self.lock = threading.Lock()
        self.thread_started = False

    def update_model(self, model):
        self.model = model
        self.ready = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Start processing thread once
        if not self.thread_started and self.ready:
            self.thread_started = True
            threading.Thread(target=self._process_loop, daemon=True).start()

        with self.lock:
            display_frame = self.output_frame if self.output_frame is not None else img

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

    def _process_loop(self):
        while True:
            if not self.ready:
                continue
            # Skip processing unless last frame exists
            try:
                with self.lock:
                    input_frame = self.output_frame

                if input_frame is not None:
                    processed_input = preprocess_frame(input_frame)
                    output = self.model.predict(processed_input, verbose=0)
                    processed_output = postprocess_frame(output)
                    processed_output = cv2.cvtColor(processed_output, cv2.COLOR_RGB2BGR)

                    with self.lock:
                        self.output_frame = processed_output

            except Exception as e:
                print("Error in processing thread:", e)




# ----------------------
# Streamlit UI
# ----------------------
st.title("📷 Real-Time Dehazing (WebRTC-enabled)")

mode = st.radio("Choose input method:", ["Webcam", "Upload Image", "Upload Video"])
location = st.selectbox("Select location:", ["Patiala", "Thapar Campus"])

model = load_model(location)

# Initialize processor in session state if not already there
if "processor" not in st.session_state:
    st.session_state.processor = VideoProcessor()
st.session_state.processor.update_model(model)

# ----------------------
# Handle Modes
# ----------------------
# --- Inside the Webcam block ---

if mode == "Webcam":
    st.info("Make sure to allow webcam access.")

    def processor_factory():
        return st.session_state.get("processor", VideoProcessor())


    webrtc_streamer(
        key="dehazing",
        video_processor_factory=processor_factory,
        media_stream_constraints={"video": True, "audio": False}
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
