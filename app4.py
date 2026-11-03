import streamlit as st
import nibabel as nib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from skimage.morphology import binary_opening, binary_closing, ball
from skimage.measure import marching_cubes
import os

st.set_page_config(page_title="NeuroScope", layout="wide")
st.title("NeuroScope: Interactive Neuroimaging & Analysis")

# Initialize session state for the experiment dashboard
if 'experiments' not in st.session_state:
    st.session_state.experiments = []

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
        value=float(np.mean(data) * 1.5)
    )
    
    apply_opening = st.sidebar.checkbox("Apply Binary Opening")
    apply_closing = st.sidebar.checkbox("Apply Binary Closing")

    # Generate computational mask
    mask = data > intensity_threshold
    
    if apply_opening or apply_closing:
        structuring_element = ball(1)
        if apply_opening:
            mask = binary_opening(mask, structuring_element)
        if apply_closing:
            mask = binary_closing(mask, structuring_element)

    # Quantitative Metrics
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    fg_voxels = int(np.sum(mask))
    phys_vol_ml = (fg_voxels * voxel_vol) / 1000

    st.sidebar.markdown("---")
    st.sidebar.subheader("Record Experiment")
    if st.sidebar.button("Save Current Configuration"):
        st.session_state.experiments.append({
            "Threshold": intensity_threshold,
            "Opening": apply_opening,
            "Closing": apply_closing,
            "Voxels": fg_voxels,
            "Volume (mL)": round(phys_vol_ml, 2)
        })
        st.sidebar.success("Recorded!")

    # Top-level metrics
    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Foreground Voxels", f"{fg_voxels:,}")
    col_met2.metric("Physical Volume (mL)", f"{phys_vol_ml:,.2f}")
    col_met3.metric("Voxel Spacing (mm)", f"{spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f}")

    def create_overlay(slice_data, slice_mask):
        norm = (slice_data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 255
        rgb = np.stack((norm,)*3, axis=-1).astype(np.uint8)
        rgb[slice_mask, 0] = 255
        rgb[slice_mask, 1] = 0
        rgb[slice_mask, 2] = 0
        return rgb

    # Setup interface tabs
    tab1, tab2, tab3 = st.tabs(["2D Multiplanar Viewer", "Research Dashboard", "3D Explorer"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Axial")
            z_idx = st.slider("Z-Axis", 0, data.shape[2]-1, data.shape[2]//2)
            fig_ax = px.imshow(create_overlay(data[:,:,z_idx].T, mask[:,:,z_idx].T), origin='lower')
            fig_ax.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_ax, width="stretch")
            
        with col2:
            st.subheader("Coronal")
            y_idx = st.slider("Y-Axis", 0, data.shape[1]-1, data.shape[1]//2)
            fig_cor = px.imshow(create_overlay(data[:,y_idx,:].T, mask[:,y_idx,:].T), origin='lower')
            fig_cor.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_cor, width="stretch")
            
        with col3:
            st.subheader("Sagittal")
            x_idx = st.slider("X-Axis", 0, data.shape[0]-1, data.shape[0]//2)
            fig_sag = px.imshow(create_overlay(data[x_idx,:,:].T, mask[x_idx,:,:].T), origin='lower')
            fig_sag.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_sag, width="stretch")

    with tab2:
        st.subheader("Parameter Sensitivity History")
        if st.session_state.experiments:
            df = pd.DataFrame(st.session_state.experiments)
            st.dataframe(df, width="stretch")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data as CSV", data=csv, file_name="neuroscope_experiment.csv", mime="text/csv")
            
            if len(df) > 1:
                fig_line = px.line(df, x=df.index, y="Volume (mL)", title="Volume Variation Across Configurations", markers=True)
                st.plotly_chart(fig_line, width="stretch")
        else:
            st.info("Adjust segmentation controls and click 'Save Current Configuration' to begin recording measurements.")

    with tab3:
        st.subheader("3D Surface Reconstruction")
        st.caption("Rendering a 3D mesh is computationally heavy. Click the button below to generate.")
        if st.button("Generate 3D Model"):
            with st.spinner("Calculating marching cubes..."):
                try:
                    verts, faces, normals, values = marching_cubes(mask, level=0.5, spacing=spacing)
                    fig_3d = go.Figure(data=[go.Mesh3d(
                        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        opacity=0.6,
                        colorscale='Reds',
                        intensity=verts[:, 2]
                    )])
                    fig_3d.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_3d, width="stretch")
                except ValueError:
                    st.error("Mask is empty. Lower the threshold to capture tissue before rendering.")
else:
    st.error(f"Waiting for data: Please input your NIfTI file at {img_path}")