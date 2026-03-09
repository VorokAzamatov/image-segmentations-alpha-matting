import torch
import cv2
import numpy as np
import streamlit as st

from pathlib import Path
from PIL import Image

from data.transforms import get_val_transforms
from inference.pipeline import load_model, predict_single_image


@st.cache_resource
def load_cached_model(model_type, weights_path, device):
    return load_model(model_type, weights_path, device=device)


def show_results(img, mask):
    ARROW_STYLE = "text-align:center;font-size:50px;margin-top:90%;"

    col1, col2, col3 = st.columns([2, 1, 2])

    col1.image(img, caption="Original image", width='stretch')
    col2.markdown(f"<div style='{ARROW_STYLE}'>→</div>", unsafe_allow_html=True,)
    col3.image(mask, caption="Predicted mask", width='stretch')

    download_image_btn(mask, col3, file_type='png')


def download_image_btn(mask, col, file_type):
    success, buffer = cv2.imencode(f".{file_type}", mask)

    if success:
        col.download_button(
            label="Download mask",
            data=buffer.tobytes(),
            file_name="segmentation_mask.png",
            mime="image/png",
            width='stretch',
        )


def apply_contours(overlay_on, countours_from):
    contours, _ = cv2.findContours(countours_from, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    epsilon_factor = st.sidebar.slider("Contour smoothing", 0.001, 0.05, step=0.001)


    smoothed_contours = []
    for cnt in contours:

        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        smoothed_contours.append(approx)

    overlay = np.array(overlay_on).copy()

    cv2.drawContours(overlay, smoothed_contours, -1, (0, 255, 0), 2)

    return overlay

@st.cache_resource
def run_segmentation(input_image, model_option, device):
    transforms = get_val_transforms()
    model_type, weights_path = model_option
    model = load_cached_model(model_type, weights_path, device=device)
    
    mask = predict_single_image(input_image, model, transforms, device=device)

    return mask


def apply_postprocessing(raw_mask):
    st.sidebar.caption("Postprocessing settings")

    mask = raw_mask.copy()

    show_orig_mask = st.sidebar.toggle("Show original mask")

    use_blur = st.sidebar.checkbox("Blur", value=False)
    if use_blur:
        gaus_blur_kernel = st.sidebar.slider("Gaussian blur kernel", 1, 99, step=2)
        mask = cv2.GaussianBlur(mask, (gaus_blur_kernel, gaus_blur_kernel), 0)

        median_blur_kernel = st.sidebar.slider("Median blur kernel", 1, 99, step=2)
        mask = cv2.medianBlur(mask, median_blur_kernel)

    use_threshold = st.sidebar.checkbox("Threshold", value=False)
    if use_threshold:
        threshold = st.sidebar.slider("Threshold", 0, 255, 1)
        mask[mask < threshold] = 0

    use_double_threshold = st.sidebar.checkbox("Double threshold", value=False)
    if use_double_threshold:
        low_thr, high_thr = st.sidebar.slider(
            "Double threshold",
            0, 255, (0, 255)
        )
        mask[mask < low_thr] = 0
        mask[mask > high_thr] = 255

    use_morphology = st.sidebar.checkbox("Morphology", value=False)
    if use_morphology:
        erode_kernel = st.sidebar.slider("Erode kernel", 1, 55, step=2)
        erode_kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
        
        dilation_kernel = st.sidebar.slider("Dilation kernel", 1, 55, step=2)
        dilation_kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
        mask = cv2.dilate(mask, dilation_kernel, iterations=1)

        opening_kernel = st.sidebar.slider("Opening kernel", 1, 55, step=2)
        opening_kernel = np.ones((opening_kernel, opening_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
        
        closing_kernel = st.sidebar.slider("Closing kernel", 1, 55, step=2)
        closing_kernel = np.ones((closing_kernel, closing_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

    if show_orig_mask:
        mask = raw_mask.copy()

    return mask




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    st.set_page_config(page_title="ML Demo", layout="wide")
    st.title("Image Segmentation & Alpha Matting", text_alignment='center')


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
    model_option = MODEL_OPTIONS[model_option] 


    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_img:

        img = Image.open(uploaded_img)

        raw_mask = run_segmentation(uploaded_img, model_option, device)
        mask = raw_mask.copy()

        st.sidebar.title("Postprocessing settings")

        use_postprocessing = st.sidebar.toggle("Postprocessing", value=False)
        if use_postprocessing:
            mask = apply_postprocessing(raw_mask)

            use_contours = st.sidebar.checkbox("Show contours (based on the mask)")
            if use_contours:
                img = apply_contours(img, mask)

        
        show_results(img, mask)

        


if __name__ == '__main__':
    main()