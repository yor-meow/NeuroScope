import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import nibabel as nib
import plotly.express as px
import plotly.graph_objects as go
from skimage.morphology import binary_opening, binary_closing, ball
from skimage.measure import marching_cubes

# Page Configuration
st.set_page_config(
    page_title="NeuroScope Workstation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("NeuroScope: Interactive Neuroimaging Workstation")
st.caption("A research prototype for 3D MRI slice navigation, quantitative volumetric analysis, parameter sensitivity tracking, and surface mesh extraction.")

# Initialize Session State for Experiment Logging
if 'experiments' not in st.session_state:
    st.session_state.experiments = []

# Sidebar: File Uploader & Local Fallback
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload NIfTI Scan (.nii / .nii.gz)", type=["nii", "gz"])

default_path = os.path.join("data", "sample_t1.nii")

img = None
data = None
spacing = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()[:3]
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded NIfTI file: {e}")
elif os.path.exists(default_path):
    try:
        img = nib.load(default_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()[:3]
        st.sidebar.info("Using local sample dataset (`data/sample_t1.nii`)")
    except Exception as e:
        st.sidebar.error(f"Error loading sample dataset: {e}")

if data is not None and spacing is not None:
    # Sidebar: Segmentation Controls
    st.sidebar.markdown("---")
    st.sidebar.header("Segmentation Controls")

    intensity_threshold = st.sidebar.slider(
        "Intensity Threshold",
        min_value=float(np.min(data)),
        max_value=float(np.max(data)),
        value=float(np.mean(data) * 1.5),
        step=1.0
    )

    apply_opening = st.sidebar.checkbox("Apply Binary Opening", help="Removes floating noise voxels outside main structures.")
    apply_closing = st.sidebar.checkbox("Apply Binary Closing", help="Fills internal holes and gaps within segmented structures.")

    # Compute Computational Binary Mask
    mask = data > intensity_threshold

    if apply_opening or apply_closing:
        structuring_element = ball(1)
        if apply_opening:
            mask = binary_opening(mask, structuring_element)
        if apply_closing:
            mask = binary_closing(mask, structuring_element)

    # Volumetric Calculations
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    fg_voxels = int(np.sum(mask))
    phys_vol_ml = (fg_voxels * voxel_vol_mm3) / 1000.0

    # Session Recording Control
    st.sidebar.markdown("---")
    st.sidebar.subheader("Record Session State")
    if st.sidebar.button("Save Current Configuration"):
        st.session_state.experiments.append({
            "Run": len(st.session_state.experiments) + 1,
            "Threshold": round(intensity_threshold, 2),
            "Opening": apply_opening,
            "Closing": apply_closing,
            "Voxels": fg_voxels,
            "Volume (mL)": round(phys_vol_ml, 2)
        })
        st.sidebar.success("Configuration recorded!")

    # Top-Level Quantitative Metrics Banner
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Foreground Voxels", f"{fg_voxels:,}")
    col_m2.metric("Physical Volume (mL)", f"{phys_vol_ml:,.2f} mL")
    col_m3.metric("Voxel Spacing (mm)", f"{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f}")

    # Custom RGB Mask Overlay Helper
    def create_overlay(slice_data, slice_mask):
        data_min, data_max = np.min(data), np.max(data)
        norm = (slice_data - data_min) / (data_max - data_min + 1e-8) * 255.0
        rgb = np.stack((norm,) * 3, axis=-1).astype(np.uint8)
        rgb[slice_mask, 0] = 255
        rgb[slice_mask, 1] = 0
        rgb[slice_mask, 2] = 0
        return rgb

    # UI Tab Layout
    tab1, tab2, tab3 = st.tabs(["2D Multiplanar Viewer", "Research Dashboard", "3D Explorer"])

    with tab1:
        st.subheader("Orthogonal Multiplanar Slice Navigation")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Axial View (Z-Axis)**")
            z_idx = st.slider("Z Slice", 0, data.shape[2] - 1, data.shape[2] // 2)
            fig_ax = px.imshow(create_overlay(data[:, :, z_idx].T, mask[:, :, z_idx].T), origin='lower')
            fig_ax.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_ax, width="stretch")

        with c2:
            st.markdown("**Coronal View (Y-Axis)**")
            y_idx = st.slider("Y Slice", 0, data.shape[1] - 1, data.shape[1] // 2)
            fig_cor = px.imshow(create_overlay(data[:, y_idx, :].T, mask[:, y_idx, :].T), origin='lower')
            fig_cor.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_cor, width="stretch")

        with c3:
            st.markdown("**Sagittal View (X-Axis)**")
            x_idx = st.slider("X Slice", 0, data.shape[0] - 1, data.shape[0] // 2)
            fig_sag = px.imshow(create_overlay(data[x_idx, :, :].T, mask[x_idx, :, :].T), origin='lower')
            fig_sag.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_sag, width="stretch")

    with tab2:
        st.subheader("Parameter Sensitivity & Experiment Logging")
        if st.session_state.experiments:
            df = pd.DataFrame(st.session_state.experiments)
            st.dataframe(df, width="stretch")

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Experiment Data as CSV",
                data=csv_data,
                file_name="neuroscope_sensitivity_analysis.csv",
                mime="text/csv"
            )

            if len(df) > 1:
                fig_line = px.line(
                    df, 
                    x="Run", 
                    y="Volume (mL)", 
                    title="Volume Variation Across Parameter Runs", 
                    markers=True
                )
                fig_line.update_layout(xaxis_title="Run Number", yaxis_title="Estimated Volume (mL)")
                st.plotly_chart(fig_line, width="stretch")
        else:
            st.info("No recorded runs yet. Adjust segmentation controls in the sidebar and click 'Save Current Configuration'.")

    with tab3:
        st.subheader("3D Surface Mesh Reconstruction")
        st.caption("Generates a continuous 3D geometric surface mesh directly from the 3D binary array using Marching Cubes.")

        if st.button("Render 3D Model"):
            with st.spinner("Extracting surface geometry via Marching Cubes..."):
                try:
                    verts, faces, normals, values = marching_cubes(mask, level=0.5, spacing=spacing)
                    fig_3d = go.Figure(data=[go.Mesh3d(
                        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        opacity=0.7,
                        colorscale='Reds',
                        intensity=verts[:, 2]
                    )])
                    fig_3d.update_layout(
                        scene=dict(aspectmode="data"),
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_3d, width="stretch")
                except ValueError:
                    st.error("Threshold too restrictive! The binary mask is empty. Lower the threshold to capture structure.")
else:
    st.warning("Please upload a `.nii` or `.nii.gz` file in the sidebar or place a sample scan at `data/sample_t1.nii`.")