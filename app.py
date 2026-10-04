import streamlit as st
import torch
import numpy as np
import io
import os
import cv2
from PIL import Image
import torchvision.transforms as T
import matplotlib.cm as cm
from models import SARColorizerGenerator
from metrics import calculate_entropy, calculate_enl, calculate_epi

st.set_page_config(
    page_title="ISRO SIH1733: GeoColor AI Suite",
    page_icon="🛰️",
    layout="wide"
)

# Dark Mode Styling
st.markdown("""
    <style>
    .main { background-color: #0b0f17; color: #f8fafc; }
    .stMetric { background-color: #131b2e; padding: 14px; border-radius: 8px; border: 1px solid #1e293b; }
    h1, h2, h3 { color: #00f2fe; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_production_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SARColorizerGenerator()
    ckpt = 'checkpoints/best_sar_colorizer.pth'
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

model, device = load_production_model()

st.title("🛰️ GeoColor AI: Physics-Guided SAR Colorization Engine")
st.caption("ISRO SIH1733 Solution | Homomorphic Log-Domain Despeckling & Bayesian Uncertainty Guardrails")

st.sidebar.header("Mission Configurations")
category = st.sidebar.selectbox("Terrain Categorization", ["urban", "grassland", "barrenland", "agri"])
mc_passes = st.sidebar.slider("Bayesian MC Uncertainty Passes (1 = Sharpest)", 1, 10, 1)
color_vibrancy = st.sidebar.slider("Color Saturation Multiplier", 1.0, 3.0, 2.0, 0.1)
uploaded_file = st.sidebar.file_uploader("Upload Sentinel-1 SAR Image (s1)", type=["png", "jpg", "tif", "jpeg"])

def structure_guided_color_fusion(sar_clean, raw_pred_rgb, vibrancy=2.0):
    """
    Fuses crisp radar structural luminance with rich, calibrated optical chromaticity.
    Preserves roads and building edges while injecting natural Sentinel-2 colors.
    """
    # 1. Convert predicted RGB into LAB color space
    pred_uint8 = np.clip(raw_pred_rgb * 255.0, 0, 255).astype(np.uint8)
    lab_pred = cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 2. Linear histogram stretch on radar signal to preserve contrast
    sar_stretched = cv2.normalize(sar_clean, None, alpha=20, beta=235, norm_type=cv2.NORM_MINMAX).astype(np.float32)

    # 3. Blend structural Luminance (70% radar details + 30% optical base)
    lab_pred[:, :, 0] = 0.70 * sar_stretched + 0.30 * lab_pred[:, :, 0]

    # 4. Boost natural chromaticity (a* and b* channels centered at 128)
    lab_pred[:, :, 1] = 128.0 + (lab_pred[:, :, 1] - 128.0) * vibrancy
    lab_pred[:, :, 2] = 128.0 + (lab_pred[:, :, 2] - 128.0) * vibrancy

    # 5. Physics Guardrail: Neutralize extreme corner reflector highlights (keep concrete/metal bright gray-white)
    corner_reflectors = sar_stretched > 215
    lab_pred[:, :, 1][corner_reflectors] = 128.0 + (lab_pred[:, :, 1][corner_reflectors] - 128.0) * 0.15
    lab_pred[:, :, 2][corner_reflectors] = 128.0 + (lab_pred[:, :, 2][corner_reflectors] - 128.0) * 0.15

    # 6. Convert back to RGB
    lab_clamped = np.clip(lab_pred, 0, 255).astype(np.uint8)
    rgb_fused = cv2.cvtColor(lab_clamped, cv2.COLOR_LAB2RGB)

    # 7. Unsharp Masking for razor-sharp edge specifications
    gaussian_blur = cv2.GaussianBlur(rgb_fused, (0, 0), 1.2)
    sharpened_rgb = cv2.addWeighted(rgb_fused, 1.35, gaussian_blur, -0.35, 0)

    return np.clip(sharpened_rgb, 0, 255).astype(np.uint8)

if uploaded_file:
    raw_pil = Image.open(uploaded_file).convert('L')
    raw_arr = np.array(raw_pil).astype(np.float32)

    resized = raw_pil.resize((256, 256), Image.Resampling.BILINEAR)
    tensor = T.ToTensor()(resized).unsqueeze(0).to(device)

    with st.spinner("Executing Physics-Informed Colorization..."):
        clean_sar, mean_rgb, uncertainty = model.predict_with_epistemic_uncertainty(tensor, passes=mc_passes)

        raw_pred_rgb = mean_rgb.squeeze(0).cpu().permute(1, 2, 0).numpy()
        clean_np = np.clip(clean_sar.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

        # Apply structural fusion with adjustable color vibrancy
        rgb_np = structure_guided_color_fusion(clean_np, raw_pred_rgb, vibrancy=color_vibrancy)

        # Generate Inferno Uncertainty Heatmap
        unc_np = uncertainty.squeeze().cpu().numpy()
        unc_norm = (unc_np - unc_np.min()) / (unc_np.max() - unc_np.min() + 1e-6)
        unc_heatmap = (cm.inferno(unc_norm)[:, :, :3] * 255.0).astype(np.uint8)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("1. Raw SAR Intensity")
        st.image(raw_pil, width="stretch", caption="Single-Pol Backscatter (s1)")
    with c2:
        st.subheader("2. Despeckled Reflectivity")
        st.image(clean_np, width="stretch", caption="Homomorphic Filtered Signal")
    with c3:
        st.subheader("3. Colorized Optical RGB")
        st.image(rgb_np, width="stretch", caption="Synthesized Multi-Spectral Map")
    with c4:
        st.subheader("4. Epistemic Uncertainty")
        st.image(unc_heatmap, width="stretch", caption="Confidence Guardrail")

    st.markdown("---")
    st.subheader("Non-Reference Radar & Optical Metrics")

    enl_val = calculate_enl(clean_np)
    epi_val = calculate_epi(raw_arr, clean_np)
    entropy_val = calculate_entropy(rgb_np)
    unc_mean = float(unc_np.mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equivalent Looks (ENL)", f"{enl_val:.2f}", "Speckle Attenuation")
    m2.metric("Edge Preservation (EPI)", f"{epi_val:.3f}", "Boundary Sharpness")
    m3.metric("Spatial Entropy", f"{entropy_val:.2f} bits", "Information Density")
    m4.metric("Hallucination Risk Level", "LOW" if unc_mean < 0.04 else "ELEVATED", f"Var: {unc_mean:.5f}")

    buf = io.BytesIO()
    Image.fromarray(rgb_np).save(buf, format='PNG')
    st.download_button("Download Colorized Output", data=buf.getvalue(), file_name=f"colorized_{category}.png")
else:
    st.info("Upload a Synthetic Aperture Radar capture (from `dataset/v_2/.../s1/`) to run the inference pipeline.")