import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Fish Classifier", layout="wide")
st.title("🐟 Fish Classifier")

# =============================
# Model directory & auto-detect
# =============================
MODEL_DIR = r"C:\Users\imaya\Desktop\saved_models"

if not os.path.isdir(MODEL_DIR):
    st.error(f"Model folder not found: {MODEL_DIR}")
    st.stop()

# Find all .h5 models and show clean names (remove _best)
model_files = {
    os.path.splitext(f)[0].replace("_best", ""): os.path.join(MODEL_DIR, f)
    for f in os.listdir(MODEL_DIR)
    if f.lower().endswith(".h5")
}

if not model_files:
    st.error("❌ No .h5 model files found in the saved_models folder.")
    st.stop()

st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Choose a model:", list(model_files.keys()))
selected_model_path = model_files[selected_model_name]


# =============================
# Load model (cached)
# =============================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

try:
    model = load_model(selected_model_path)
    st.sidebar.success(f"✅ Loaded: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Failed to load model:\n{e}")
    st.stop()

# =============================
# Class names (edit if needed)
# =============================
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Validate class name length vs model output
num_out = model.output_shape[-1]
if len(class_names) != num_out:
    st.warning(
        f"Class list length ({len(class_names)}) doesn't match model output ({num_out}). "
        "Using generic names instead."
    )
    class_names = [f"class_{i}" for i in range(num_out)]

# =============================
# Determine input size dynamically
# =============================
# Most Keras models expose input_shape like (None, H, W, 3)
H, W = 224, 224  # sane default
try:
    ishape = model.input_shape
    if isinstance(ishape, list):  # some models return list for multi-input
        ishape = ishape[0]
    if len(ishape) >= 3 and ishape[1] and ishape[2]:
        H, W = int(ishape[1]), int(ishape[2])
    else:
        # fallback heuristic
        if "inception" in selected_model_name.lower():
            H, W = 299, 299
        else:
            H, W = 224, 224
except Exception:
    if "inception" in selected_model_name.lower():
        H, W = 299, 299

st.sidebar.write(f"🧩 Model input size: {H}×{W}")

# =============================
# Image upload
# =============================
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read & show image
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess
    img_resized = image.resize((W, H))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    if img_arr.ndim == 2:  # grayscale safety
        img_arr = np.stack([img_arr]*3, axis=-1)
    img_batch = np.expand_dims(img_arr, axis=0)

    # Predict
    try:
        preds = model.predict(img_batch)
        probs = preds[0]
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.stop()

    # Sort top-3
    top_idx = np.argsort(probs)[::-1]
    top3 = top_idx[:3]
    main_idx = top3[0]
    main_label = class_names[main_idx]
    main_conf = float(probs[main_idx]) * 100.0

    with col2:
        st.subheader("Main Prediction")
        st.markdown(f"**Species:** {main_label}")
        st.markdown(f"**Confidence:** {main_conf:.2f}%")

    with col3:
        st.subheader("Top 3 Predictions")
        df_top3 = pd.DataFrame({
            "Species": [class_names[i] for i in top3],
            "Confidence (%)": [float(probs[i]) * 100.0 for i in top3]
        })
        st.table(df_top3)

        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(df_top3["Species"], df_top3["Confidence (%)"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Top 3 Predictions")
        for i, v in enumerate(df_top3["Confidence (%)"]):
            ax.text(v + 1, i, f"{v:.2f}%", va='center')
        st.pyplot(fig)

    # Optional: show raw probabilities (expandable)
    with st.expander("Show all class probabilities"):
        full_df = pd.DataFrame({
            "Species": class_names,
            "Confidence (%)": [float(p)*100.0 for p in probs]
        }).sort_values("Confidence (%)", ascending=False).reset_index(drop=True)
        st.dataframe(full_df, use_container_width=True)
else:
    st.info("👆 Upload a JPG/PNG image to get predictions.")