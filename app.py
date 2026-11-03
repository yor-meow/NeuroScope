import streamlit as st
import nibabel as nib
import plotly.express as px
import os

st.set_page_config(page_title="NeuroScope", layout="wide")
st.title("NeuroScope: Day 1")

img_path = os.path.join("data", "sample_t1.nii")

if os.path.exists(img_path):
    img = nib.load(img_path)
    data = img.get_fdata()
    hdr = img.header

    st.subheader("Image Metadata")
    col1, col2 = st.columns(2)
    col1.write(f"**Array Shape:** {data.shape}")
    col2.write(f"**Voxel Spacing (mm):** {hdr.get_zooms()[:3]}")

    mid_slice_idx = data.shape[2] // 2
    slice_data = data[:, :, mid_slice_idx]

    st.subheader("Axial View")
    fig = px.imshow(
        slice_data.T,
        color_continuous_scale='gray',
        origin='lower',
        title=f"Slice Index: {mid_slice_idx}"
    )

    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"Waiting for data: Please input your NIfTI file at {img_path}")