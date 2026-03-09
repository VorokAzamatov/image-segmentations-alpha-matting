import torch
import cv2
import numpy as np
import streamlit as st

from pathlib import Path
from PIL import Image

from data.transforms import get_val_transforms
from inference.pipeline import load_model, predict_single_image


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="ML Demo", layout="wide")
st.title("Image Segmentation & Alpha Matting", text_alignment='center')


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

st.sidebar.title("Postprocessing settings")
use_postprocessing = st.sidebar.checkbox("Postprocessing", value=False)

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

    img = Image.open(uploaded_img)
    raw_mask = predict_single_image(uploaded_img, model, transforms, device=device)

    mask = raw_mask.copy()
    if use_postprocessing:
        use_blur = st.sidebar.checkbox("Blur", value=False)
        if use_blur:
            gaus_blur_kernel = st.sidebar.slider("Gaussian blur intensity", 1, 25, step=2)
            if gaus_blur_kernel > 1:
                mask = cv2.GaussianBlur(mask, (gaus_blur_kernel, gaus_blur_kernel), 0)

            median_blur_kernel = st.sidebar.slider("Median blur intensity", 1, 99, step=2)
            if median_blur_kernel > 1:
                mask = median_blur = cv2.medianBlur(mask, median_blur_kernel)

        use_threshold = st.sidebar.checkbox("Threshold", value=False)
        if use_threshold:
            threshold = st.sidebar.slider("Threshold", 0, 255, 1)
            mask[mask < threshold] = 0

        use_double_threshold = st.sidebar.checkbox("Double threshold", value=False)
        if use_double_threshold:
            low_thr = st.sidebar.slider("Low threshold", 0, 255, 1)
            high_thr = st.sidebar.slider("High threshold", 0, 255, 1)  
            mask[mask < low_thr] = 0
            mask[mask > high_thr] = 255

        
        
    col1, col2, col3 = st.columns([2, 1, 2])

    col1.image(img, caption="Original image", width='stretch')
    col2.markdown(f"<div style='{arrow_style}'>→</div>", unsafe_allow_html=True,)
    col3.image(mask, caption="Predicted mask", width='stretch')
    

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
            width='stretch',
        )