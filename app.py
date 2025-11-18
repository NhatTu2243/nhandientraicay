import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== Cấu hình cơ bản =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs_multi" / "fruit_model.keras"
CLASS_PATH = BASE_DIR / "outputs_multi" / "class_indices.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Nhận diện 14 loại trái cây", layout="wide")


@st.cache_resource
def load_model_and_classes():
    # Load model (có Lambda preprocess_input bên trong)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )
    # Đọc mapping index -> tên lớp
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)  # {"0": "cachua", ...}
    idx_to_class = {int(k): v for k, v in class_map.items()}
    return model, idx_to_class


model, idx_to_class = load_model_and_classes()

# ===== Giao diện =====
st.title("🍎🍌🍊 nhận diện 14 loại trái cây")

uploaded_files = st.file_uploader(
    "Chọn một hoặc nhiều ảnh trái cây",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    cols_per_row = 3  # số ảnh mỗi hàng

    for i in range(0, len(uploaded_files), cols_per_row):
        row_files = uploaded_files[i : i + cols_per_row]
        cols = st.columns(len(row_files))

        for file, col in zip(row_files, cols):
            with col:
                # Đọc & hiển thị ảnh (nhỏ lại theo ô cột)
                img = Image.open(file).convert("RGB")
                st.image(img, caption=file.name, width=350)

                # Chuẩn bị input cho model
                img_resized = img.resize(IMG_SIZE)
                x = np.array(img_resized, dtype=np.float32)
                x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
                # KHÔNG preprocess_input ở đây nữa, model tự xử lý

                # Dự đoán
                preds = model.predict(x, verbose=0)
                probs = preds[0]
                top_idx = int(np.argmax(probs))
                top_label = idx_to_class[top_idx]
                top_prob = float(probs[top_idx])

                st.markdown(f"**Kết quả:** {top_label} ({top_prob:.3f})")

                # Hiển thị top-3
                top3_idx = np.argsort(probs)[::-1][:3]
                top3_text = " / ".join(
                    f"{idx_to_class[int(j)]}: {probs[j]:.2f}"
                    for j in top3_idx
                )
                st.caption(f"Top-3: {top3_text}")


