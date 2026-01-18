import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Vitamin Deficiency Detection",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Vitamin Deficiency Detection")
st.caption("Demo only. Not a medical diagnosis.")

# -----------------------------
# LOAD MODEL (ONCE)
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model(
        "model_saved_files/others/CNN.h5",
        compile=False
    )

model = load_trained_model()

# -----------------------------
# CLASS LABELS (FIXED ORDER)
# -----------------------------
classes = [
    "aloperia areata",
    "beaus lines",
    "bluish nail",
    "bulging eyes",
    "cataracts eyes",
    "clubbing",
    "crossed eyes",
    "Dariers disease",
    "eczema",
    "glucoma eyes",
    "Lindsays nails",
    "lip",
    "tounge",
    "normal"
]

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (lips / nails / tongue / eyes)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # PREPROCESS IMAGE
    # -----------------------------
    h, w = model.input_shape[1], model.input_shape[2]
    img = img.resize((h, w))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # PREDICT
    # -----------------------------
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    preds = classes[idx]
    confidence = prediction[0][idx]

    # -----------------------------
    # VITAMIN MESSAGE LOGIC
    # -----------------------------
    if preds == "aloperia areata":
        msg = "VITAMIN DEFICIENCY-D"
        msg1 = "Mediterranean diet, fruits, vegetables, nuts, whole grains."
    elif preds == "beaus lines":
        msg = "VITAMIN DEFICIENCY-C"
        msg1 = "Leafy greens, quinoa, almonds, magnesium-rich foods."
    elif preds == "bluish nail":
        msg = "VITAMIN DEFICIENCY-B12"
        msg1 = "Eggs, fish, leafy greens, beans, whole grains."
    elif preds == "bulging eyes":
        msg = "VITAMIN DEFICIENCY-A"
        msg1 = "Bananas, yogurt, potatoes, dried apricots."
    elif preds == "lip":
        msg = "VITAMIN DEFICIENCY-B2"
        msg1 = "Milk, eggs, carrots, spinach, apricots."
    elif preds == "tounge":
        msg = "VITAMIN DEFICIENCY-B3"
        msg1 = "Yogurt, hydration, balanced diet."
    else:
        msg = "VITAMIN DEFICIENCY-D"
        msg1 = "Balanced diet with fruits, vegetables, eggs, nuts, fish."

    # -----------------------------
    # DISPLAY RESULT
    # -----------------------------
    st.subheader("Prediction Result")
    st.write(f"**Condition:** {preds}")
    st.write(f"**{msg}**")
    st.progress(float(confidence))
    st.caption(f"Confidence: {confidence:.2%}")
    st.write(msg1)
