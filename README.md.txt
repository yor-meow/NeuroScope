# NeuroScope: Interactive Neuroimaging and Segmentation Sensitivity Platform

NeuroScope is a research-oriented biomedical imaging platform built in Python. It provides interactive 3D NIfTI MRI volume navigation, threshold and morphological segmentation, quantitative volumetric analysis, parameter sensitivity logging, and 3D surface mesh generation.

## Research Question

How sensitive are brain MRI segmentation outputs and estimated region volumes to image-processing parameter variations?

## Key Features

- Multiplanar Orthogonal Navigation: Synchronized 2D Axial, Coronal, and Sagittal slice viewers using NiBabel and Plotly.
- Morphological Processing: 3D binary opening and closing operations via scikit-image.
- Quantitative Metrics: Direct conversion of 3D array voxel counts into physical volumes (mm3 and mL) using NIfTI header spacing metadata.
- Reproducibility Engine: Live session tracking, parameter sensitivity plotting, and CSV export.
- 3D Geometric Extraction: Marching Cubes surface mesh reconstruction rendered via Plotly Mesh3d.

## Project Structure

NeuroScope/
├── app4.py
├── pytest.ini
├── requirements.txt
├── README.md
├── data/
│   └── sample_t1.nii
├── neuroscope/
│   ├── __init__.py
│   └── analysis.py
└── tests/
    └── test_analysis.py

## Installation

1. Clone the repository and navigate to the project directory:
git clone https://github.com/your-username/NeuroScope.git
cd NeuroScope

2. Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

## Running the Application

Launch the application using the Streamlit runtime:
streamlit run app4.py

## Testing

Run unit tests for numerical and volumetric functions:
python -m pytest