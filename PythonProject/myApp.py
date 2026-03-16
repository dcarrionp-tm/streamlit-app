import os
# This environment variable MUST be set before importing tensorflow/keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# Use tf_keras for compatibility with older .h5 files
try:
    from tf_keras.models import load_model
except ImportError:
    st.error("Please run: pip install tf-keras")
    st.stop()

st.title("Teachable Machine Image Model")

st.markdown("""
Upload an image to get a prediction from your Teachable Machine model.
""")

@st.cache_resource
@st.cache_resource
def get_model():
    try:
        # Get the directory where myApp.py is actually located
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, "keras_model.h5")
        labels_path = os.path.join(base_path, "labels.txt")

        # Load the model using the calculated path
        model = load_model(model_path, compile=False)

        # Load the labels
        with open(labels_path, "r") as f:
            class_names = f.readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, class_names = get_model()

if model is None:
    st.warning("Please place `keras_Model.h5` and `labels.txt` in the same directory as this script.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Prepare the image for the model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        image_array = np.asarray(image)
        # Normalize: Teachable Machine expects -1 to 1 range
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Run inference
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        st.subheader("Results:")
        # Display name (stripping the index prefix like '0 ')
        st.write(f"**Class:** {class_name[2:].strip()}")
        st.write(f"**Confidence Score:** {confidence_score:.2%}")