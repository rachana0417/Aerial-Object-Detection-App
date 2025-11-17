import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tensorflow as tf

# Load YOLO model
yolo_model = YOLO("runs/detect/train/weights/best.pt")

# Load MobileNet model (for classification - bird/drone)
mobilenet_model = tf.keras.models.load_model("best_mobilenet_finetuned.keras")

CLASS_NAMES = ["Bird", "Drone"]

st.title("Aerial Object Detection App")
st.write("Upload an image to detect **Bird** or **Drone**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # YOLO detection
    st.subheader("YOLO Detection Result")
    results = yolo_model(img)
    results[0].plot()
    st.image(results[0].plot(), caption="Detection Output", use_column_width=True)

    # Classification using MobileNet
    st.subheader("Classification Result")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = mobilenet_model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"**Predicted Class:** {CLASS_NAMES[class_id]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
