import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
import os
import threading
import time
import cv2

# ----------------------------
# Model info: Google Drive file IDs and filenames
# ----------------------------
MODEL_INFO = {
    "Patiala": {
        "file_id": "19hbgk6afotZ6rt_LV9y8qK7Jiw3kWRbQ",
        "filename": "mixed_noaug.keras",
    },
    "Thapar Campus": {
        "file_id": "1HIwQqPoZShblcuG4Sc2kJ5qoi4w18ZTS",
        "filename": "pix2pix.keras",
    },
}

# ----------------------------
# Load model from cloud and cache it
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_cloud_model(location):
    info = MODEL_INFO[location]
    if not os.path.exists(info["filename"]):
        url = f"https://drive.google.com/uc?id={info['file_id']}"
        gdown.download(url, info["filename"], quiet=False)
    model = tf.keras.models.load_model(info["filename"])
    return model

# ----------------------------
# Frame preprocess/postprocess
# ----------------------------
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 127.5 - 1
    return np.expand_dims(img, axis=0)

def postprocess_frame(output):
    img = (output[0] + 1) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ----------------------------
# Video capture thread using OpenCV for USB webcam
# ----------------------------
class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Frame read failed, retrying...")
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ----------------------------
# Streamlit app main function
# ----------------------------
def main():
    st.title("🟢 Jetson USB Webcam Real-Time Dehazing with Model Selection")

    # Model selection dropdown with unique key
    location = st.selectbox("Select Location / Model", options=list(MODEL_INFO.keys()), key="model_select")

    # Load selected model (cached)
    model = load_cloud_model(location)

    # Checkbox to start/stop webcam dehazing with unique key
    run = st.checkbox("Start Webcam Dehazing", key="start_dehaze_checkbox")

    if run:
        # Initialize video capture thread once and store in session_state
        if "video_thread" not in st.session_state:
            try:
                st.session_state.video_thread = VideoCaptureThread(src=0)  # USB webcam device index 0
            except Exception as e:
                st.error(f"Camera error: {e}")
                return

        frame_placeholder = st.empty()

        stop_signal = False
        while run and not stop_signal:
            frame = st.session_state.video_thread.read()
            if frame is not None:
                input_tensor = preprocess_frame(frame)
                output = model.predict(input_tensor)
                dehazed = postprocess_frame(output)

                combined = np.hstack((
                    cv2.resize(frame, (256, 256)),
                    cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
                ))

                frame_placeholder.image(combined, channels="BGR")
            else:
                st.write("Waiting for camera frame...")

            time.sleep(0.03)

            # Check checkbox state again to allow stopping
            run = st.session_state.get("start_dehaze_checkbox", False)
            if not run:
                stop_signal = True

        # Stop capture thread on exit
        if "video_thread" in st.session_state:
            st.session_state.video_thread.stop()
            del st.session_state.video_thread
        st.write("Stopped webcam.")

    else:
        st.write("Check the box above to start webcam dehazing.")

if __name__ == "__main__":
    main()
