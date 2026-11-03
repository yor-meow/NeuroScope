import streamlit as st
import nibabel as nib
import plotly.express as px
import os

st.set_page_config(page_title="NeuroScope", layout="wide")
st.title("NeuroScope: Multiplannar Viewer")

img_path = os.path.join("data", "sample_t1.nii")

if os.path.exists(img_path):
    img = nib.load(img_path)
    data = img.get_fdata()
    hdr = img.header
    
    st.caption (f"Volume Dimensions: {data.shape} | Voxel Spacing: {hdr.get_zooms()[:3]} mm")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Axial")
        z_max = data.shape [2] - 1
        z_idx = st.slider("Superior-Inferior", 0, z_max, z_max // 2)
        fig_axial = px.imshow(data[:, :, z_idx].T, color_continuous_scale='gray', origin='lower')
        fig_axial.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_axial, use_container_width=True)

    with col2:
        st.subheader("Coronal")
        y_max = data.shape [1] - 1
        y_idx = st.slider("Anterior-Posterior, 0 y_max, y_max // 2")
        fig_coronal = px.imshow(data[:, y_idx, :].T, color_continuous_scale='gray', origin='lower')
        fig_coronal.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_coronal, use_container_width=True)
        
    with col3:
        st.subheader("Sagittal")
        x_max = data.shape[0] - 1
        x_idx = st.slider("Left-Right", 0, x_max, x_max // 2)
        fig_sagittal = px.imshow(data[x_idx, :, :].T, color_continuous_scale='gray', origin='lower')
        fig_sagittal.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_sagittal, use_container_width=True)
else:
    st.error(f"Waiting for data: Please input your NIfTI file at {img_path}")