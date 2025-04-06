import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import os
import threading

# ----------------------
# Model Loading
# ----------------------
@st.cache_resource
def load_model(location):
    if location == "Patiala":
        file_id = "19hbgk6afotZ6rt_LV9y8qK7Jiw3kWRbQ"
        filename = "mixed_noaug.keras"
    else:
        file_id = "1HIwQqPoZShblcuG4Sc2kJ5qoi4w18ZTS"
        filename = "pix2pix.keras"

    # Ensure model is downloaded correctly
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    try:
        return tf.keras.models.load_model(filename)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ----------------------
# Video Processing
# ----------------------
class DehazingProcessor(VideoProcessorBase):
    def __init__(self):
        self._model_lock = threading.Lock()
        self._model = None
        self._frame_counter = 0

    def update_model(self, model):
        with self._model_lock:
            self._model = model

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            with self._model_lock:
                if self._model:
                    # Process every frame
                    input_tensor = cv2.resize(img, (256, 256))
                    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
                    input_tensor = (input_tensor.astype(np.float32) / 127.5) - 1.0
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                    
                    output = self._model.predict(input_tensor)
                    
                    # Post-process output
                    output_frame = (output[0] + 1) * 127.5
                    output_frame = np.clip(output_frame, 0, 255).astype(np.uint8)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    
                    return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

        except Exception as e:
            st.error(f"Frame processing error: {str(e)}")
        
        return frame  # Fallback to original frame

# ----------------------
# Streamlit UI
# ----------------------
st.title("🌥️ Real-Time Cloud Dehazing")

# Initialize session state
if "processor" not in st.session_state:
    st.session_state.processor = DehazingProcessor()

# Model loading
location = st.selectbox("Select location model:", ["Patiala", "Thapar Campus"])
model = load_model(location)

if model is None:
    st.stop()

# Update processor with new model
st.session_state.processor.update_model(model)

# WebRTC Configuration
RTC_CONFIG = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": "stun:global.stun.twilio.com:3478?transport=udp"}
    ]
}

mode = st.radio("Input Mode:", ["Webcam", "Image Upload"])

if mode == "Webcam":
    st.info("Allow camera access when prompted")
    webrtc_streamer(
        key="dehazing",
        video_processor_factory=lambda: st.session_state.processor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 24}
            },
            "audio": False
        },
        async_processing=True
    )

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload hazy image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process image
        input_tensor = cv2.resize(img, (256, 256))
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        input_tensor = (input_tensor.astype(np.float32) / 127.5) - 1.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        output = model.predict(input_tensor)
        output_frame = (output[0] + 1) * 127.5
        output_frame = np.clip(output_frame, 0, 255).astype(np.uint8)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        # Display comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="Original Image")
        with col2:
            st.image(output_frame, channels="BGR", caption="Dehazed Image")
