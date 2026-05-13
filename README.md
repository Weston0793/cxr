# CXR Anatomy Segmentation Streamlit App

This project provides an inference-only Streamlit UI for
`ConstantinSeibold/ChestXRayAnatomySegmentation` through the `cxas` package.

## Features

- Upload CXR images in `png`, `jpg`, `jpeg`, and `dcm`.
- Run segmentation with `from cxas import CXAS`.
- Group CXAS anatomy classes into:
  - Airways + lungs
  - Cardiovascular / mediastinal
  - Skeletal
- View original image + three overlays + optional combined overlay.
- Download each grouped binary mask as a PNG.
- CPU by default with optional GPU selection in sidebar.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- This app performs **inference only** (no training).
- The model is cached with `st.cache_resource` to avoid repeated loads.
- Uploaded files are written into temporary folders during processing.


## Deployment note

If deployment errors mention `libGL.so.1`, ensure `opencv-python-headless` is installed (it is included in `requirements.txt`).
