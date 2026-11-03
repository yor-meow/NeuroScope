import streamlit as st
import nibabel as nib
import numpy as np
import plotly.express as px
from skimage.morphology import binary_opening, binary_closing, ball
import os

st.set_page_config(page_title="NeuroScope", layout="wide")
st.title("NeuroScope: Segmentation & Quantitative Metrics")

img_path = os.path.join("data", "sample_t1.nii")

if os.path.exists(img_path):
    img = nib.load(img_path)
    data = img.get_fdata()
    hdr = img.header
    spacing = hdr.get_zooms()[:3]
    
    st.sidebar.header("Segmentation Controls")
    intensity_threshold = st.sidebar.slider(
        "Intensity Threshold", 
        min_value=float(np.min(data)), 
        max_value=float(np.max(data)), 
        value=float(np.mean(data))
    )
    
    apply_opening = st.sidebar.checkbox("Apply Binary Opening (Remove Noise)")
    apply_closing = st.sidebar.checkbox("Apply Binary Closing (Fill Gaps)")

    mask = data > intensity_threshold
    
    if apply_opening or apply_closing:
        structuring_element = ball(1)
        if apply_opening:
            mask = binary_opening(mask, structuring_element)
        if apply_closing:
            mask = binary_closing(mask, structuring_element)

    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    foreground_voxels = np.sum(mask)
    physical_volume_mm3 = foreground_voxels * voxel_volume_mm3
    physical_volume_ml = physical_volume_mm3 / 1000

    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Foreground Voxels", f"{foreground_voxels:,}")
    col_met2.metric("Physical Volume (mm³)", f"{physical_volume_mm3:,.2f}")
    col_met3.metric("Estimated Volume (mL)", f"{physical_volume_ml:,.2f}")

    def create_overlay(slice_data, slice_mask):
        norm = (slice_data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 255
        rgb = np.stack((norm,)*3, axis=-1).astype(np.uint8)
        rgb[slice_mask, 0] = 255 
        rgb[slice_mask, 1] = 0
        rgb[slice_mask, 2] = 0
        return rgb

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Axial")
        z_idx = st.slider("Z-Axis", 0, data.shape[2]-1, data.shape[2]//2)
        fig_ax = px.imshow(create_overlay(data[:,:,z_idx].T, mask[:,:,z_idx].T), origin='lower')
        fig_ax.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_ax, use_container_width=True)
        
    with col2:
        st.subheader("Coronal")
        y_idx = st.slider("Y-Axis", 0, data.shape[1]-1, data.shape[1]//2)
        fig_cor = px.imshow(create_overlay(data[:,y_idx,:].T, mask[:,y_idx,:].T), origin='lower')
        fig_cor.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_cor, use_container_width=True)
        
    with col3:
        st.subheader("Sagittal")
        x_idx = st.slider("X-Axis", 0, data.shape[0]-1, data.shape[0]//2)
        fig_sag = px.imshow(create_overlay(data[x_idx,:,:].T, mask[x_idx,:,:].T), origin='lower')
        fig_sag.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_sag, use_container_width=True)
else:
    st.error(f"Waiting for data: Please input your NIfTI file at {img_path}")