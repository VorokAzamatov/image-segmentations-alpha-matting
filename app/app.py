import torch
import cv2
import streamlit as st

from pathlib import Path
from PIL import Image

from data.transforms import get_val_transforms
from inference.pipeline import load_model, predict_single_image


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="ML Demo", layout="wide")
st.title("Image Segmentation & Alpha Matting")


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Model settings")

MODEL_OPTIONS = {
    "Custom Unet | Pretrain": (
        "custom_unet",
        Path("artifacts/best_models/train/CUSTOMUnet_best_model.pt"),
    ),
    "Custom Unet | Finetune": (
        "custom_unet",
        Path("artifacts/best_models/finetune/FT_CUSTOMUnet_best_model.pt"),
    ),
    "SMP Unet | Pretrain": (
        "smp_unet",
        Path("artifacts/best_models/train/SMPUnet_best_model.pt"),
    ),
    "SMP Unet | Finetune": (
        "smp_unet",
        Path("artifacts/best_models/finetune/FT_SMPUnet_best_model.pt"),
    ),
}

model_option = st.sidebar.selectbox("Select model", list(MODEL_OPTIONS.keys()))


# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# MODEL CACHE (ВАЖНО ДЛЯ STREAMLIT)
# -------------------------------------------------
@st.cache_resource
def load_cached_model(model_type, weights_path):
    return load_model(model_type, weights_path, device=device)


# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "png"])


if uploaded_img:

    transforms = get_val_transforms()

    model_type, weights_path = MODEL_OPTIONS[model_option]
    model = load_cached_model(model_type, weights_path)

    # -------------------------------------------------
    # UI: ORIGINAL IMAGE
    # -------------------------------------------------
    arrow_style = (
        "text-align:center;"
        "font-size:50px;"
        "margin-top:90%;"
    )

    col1, col2, col3 = st.columns([2, 1, 2])

    img = Image.open(uploaded_img)

    col1.image(img, caption="Original image", use_container_width=True)

    col2.markdown(
        f"<div style='{arrow_style}'>→</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    _, btn_col, _ = st.columns([3, 1, 3])

    if btn_col.button("Predict", use_container_width=True):

        mask = predict_single_image(
            uploaded_img,
            model,
            transforms,
            device=device,
        )

        col3.image(mask, caption="Predicted mask", use_container_width=True)

        # -------------------------------------------------
        # DOWNLOAD RESULT
        # -------------------------------------------------
        success, buffer = cv2.imencode(".png", mask)

        if success:
            col3.download_button(
                label="Download mask",
                data=buffer.tobytes(),
                file_name="segmentation_mask.png",
                mime="image/png",
                use_container_width=True,
            )