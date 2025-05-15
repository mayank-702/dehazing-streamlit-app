import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import gdown
import os
import threading
import time

# ----------------------------
# Model info: Google Drive file IDs and filenames
# ----------------------------
MODEL_INFO = {
    "Patiala": {
        "file_id": "1l_ocEtBK2mtejCeYH0rh0eVH2GiV24Of",
        "filename": "model1.h5",
    },
    "Thapar Campus": {
        "file_id": "1mJO0ZcnMyPWq0ZbDb04AElnhLws3fUqk",
        "filename": "model2.h5",
    },
    "Model 3": {
        "file_id": "1PDZcOGlij0jEAq7761G4eHB4PjT-Xzpq",
        "filename": "model3.h5",
    },
    "Model 4": {
        "file_id": "1U3GH5b4F9PrNAINg5ChIjFcN8zLGg56Z",
        "filename": "model4.h5",
    },
    "Model 5": {
        "file_id": "1Q343NNIf2ZemgwhegGPfIT3eC0r-jODf",
        "filename": "model5.h5",
    }
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
# Background video capture thread
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
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ----------------------------
# Stack images horizontally with captions
# ----------------------------
def stack_images_with_captions(input_frame, outputs_dict, labels):
    # Resize input frame to 256x256 for alignment
    input_resized = cv2.resize(input_frame, (256, 256))
    input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)

    imgs = [input_rgb]  # start with input
    for label in labels[1:]:
        imgs.append(outputs_dict[label])  # outputs are already RGB 256x256

    # Add captions as black bars above each image
    caption_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    bg_color = (30, 30, 30)

    captioned_imgs = []
    for img, label in zip(imgs, labels):
        caption_bar = np.full((caption_height, img.shape[1], 3), bg_color, dtype=np.uint8)
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (caption_height + text_size[1]) // 2
        cv2.putText(caption_bar, label, (text_x, text_y), font, font_scale, font_color, 2, cv2.LINE_AA)
        combined = np.vstack((caption_bar, img))
        captioned_imgs.append(combined)

    # Stack all images horizontally
    stacked = np.hstack(captioned_imgs)
    return stacked

# ----------------------------
# Streamlit app main function
# ----------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("🟢 Jetson AGX Orin - Real-Time Dehazing with Model Selection")

    # Load all models once
    with st.spinner("Loading models..."):
        models = {loc: load_cloud_model(loc) for loc in MODEL_INFO.keys()}
    st.success("✅ Models loaded!")

    # Dropdown for best model (optional selection) - placed at top
    selected_model_name = st.selectbox(
        "👉🏼 Select the best model (optional):",
        [""] + list(MODEL_INFO.keys()),
        key="select_best_model_dropdown"
    )

    # Start webcam checkbox
    run = st.checkbox("Start Webcam", key="start_webcam")

    if run:
        if "video_thread" not in st.session_state:
            try:
                st.session_state.video_thread = VideoCaptureThread()
            except RuntimeError as e:
                st.error(f"Camera error: {e}")
                return

        frame_placeholder = st.empty()
        labels_all = ["Input"] + list(MODEL_INFO.keys())

        while st.session_state.get("start_webcam", False):
            frame = st.session_state.video_thread.read()
            if frame is None:
                st.warning("Waiting for camera...")
                time.sleep(0.03)
                continue

            input_tensor = preprocess_frame(frame)

            if selected_model_name == "":
                # No model selected: show input + all model outputs stacked horizontally
                outputs = {}
                for name, model in models.items():
                    pred = model.predict(input_tensor, verbose=0)
                    outputs[name] = postprocess_frame(pred)  # RGB 256x256

                combined_img = stack_images_with_captions(frame, outputs, labels_all)  # RGB combined image
                frame_placeholder.image(combined_img, channels="RGB", use_container_width=True)

            else:
                # Model selected: show input and selected model output side-by-side
                model = models[selected_model_name]
                output = model.predict(input_tensor, verbose=0)
                output_img = postprocess_frame(output)  # RGB 256x256

                input_resized = cv2.resize(frame, (256, 256))
                input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)

                # Stack input and output vertically with captions
                combined_img = stack_images_with_captions(
                    frame,
                    {selected_model_name: output_img},
                    ["Input", selected_model_name]
                )
                frame_placeholder.image(combined_img, channels="RGB", use_container_width=True)

            time.sleep(0.03)

        # Stop camera on uncheck
        if "video_thread" in st.session_state:
            st.session_state.video_thread.stop()
            del st.session_state.video_thread
        st.write("Webcam stopped.")

    else:
        st.write("☝️ Please check the box to start webcam.")

if __name__ == "__main__":
    main()
