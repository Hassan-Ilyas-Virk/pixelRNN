import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import numpy as np
import cv2
import os

# Import model definitions from train.py
from train import PixelRNN


# =====================================================
# MASK CREATION — precise black box detection
# =====================================================
def create_black_box_mask(pil_image, threshold=15):
    """
    Detects pure/near-black regions as occlusions.
    Uses OpenCV for sharp mask boundaries.
    """
    np_img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Threshold for near-black (tuneable)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup (remove small dots)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)

    mask = binary.astype(np.float32) / 255.0
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    return mask


# =====================================================
# INPAINT FUNCTION
# =====================================================
def inpaint_image(model, pil_image, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    img = pil_image.convert("RGB")
    tensor_img = transform(img).unsqueeze(0).to(device)
    mask = create_black_box_mask(pil_image).to(device)

    mask = F.interpolate(mask, size=(128, 128), mode="nearest")

    with torch.no_grad():
        pred = model(tensor_img, mask)
        comp = tensor_img * (1 - mask) + pred * mask

    out_img = transforms.ToPILImage()(comp.squeeze(0).cpu())
    return out_img, transforms.ToPILImage()(mask.squeeze(0).cpu())


# =====================================================
# STREAMLIT APP
# =====================================================
st.title("PixelRNN Image Inpainting")
st.caption("Upload an occluded image — the model will fill in only the black box regions.")

uploaded_file = st.file_uploader("Upload your occluded image:", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption=" Uploaded Image", width=200)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PixelRNN(in_channels=4, hidden_channels=48, num_layers=3).to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch190.pth", map_location=device))
    model.eval()

    with st.spinner("Inpainting in progress..."):
        completed_image, mask_image = inpaint_image(model, image, device)

    st.subheader("Completed Image")
    st.image(completed_image, caption="Inpainted Result", width=200)

    # Optional download button
    st.download_button(
        label="Download Completed Image",
        data=completed_image.tobytes(),
        file_name="completed_image.png",
        mime="image/png"
    )
