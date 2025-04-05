import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# Mapping of location to Google Drive model file IDs
import gdown


def download_model(location):
    if location == "Patiala":
        url = "https://drive.google.com/uc?id=19hbgk6afotZ6rt_LV9y8qK7Jiw3kWRbQ"
        filename = "mixed_noaug.keras"
    elif location == "Thapar Campus":
        url = "https://drive.google.com/uc?id=1HIwQqPoZShblcuG4Sc2kJ5qoi4w18ZTS"
        filename = "pix2pix.keras"
    else:
        return None

    gdown.download(url, filename, quiet=False)
    return filename


# Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))
    img = np.array(image).astype(np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img

# Postprocessing function
def postprocess_image(predicted):
    output = (predicted[0] + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# Streamlit UI
st.title("🌫️ Real-Time Dehazing App")

# Model location dropdown
location = st.selectbox("Choose location", ["Patiala", "Thapar Campus"])

# Load model
model_file = download_model(location)
model = tf.keras.models.load_model(model_file)
st.success(f"Model loaded: {model_file}")

# Input mode selection
mode = st.radio("Select input mode", ["📷 Webcam", "🎥 Upload Video", "🖼️ Upload Image"])

# Exit control (simulate by using session state)
if "stop" not in st.session_state:
    st.session_state["stop"] = False

if st.button("❌ Exit"):
    st.session_state["stop"] = True

if not st.session_state["stop"]:

    if mode == "📷 Webcam":
        st.info("Webcam support only works locally or with `streamlit-webrtc` integration.")
        st.warning("Run locally to test webcam support or use video/image input here on cloud.")

    elif mode == "🎥 Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = open("temp_video.mp4", 'wb')
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture("temp_video.mp4")
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                input_img = preprocess_image(pil_img)
                output = model.predict(input_img)
                out_img = postprocess_image(output)

                stframe.image(out_img, channels="RGB", use_column_width=True)

            cap.release()

    elif mode == "🖼️ Upload Image":
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)

            input_img = preprocess_image(image)
            output = model.predict(input_img)
            out_img = postprocess_image(output)

            st.image(out_img, caption="Dehazed Image", use_column_width=True)
