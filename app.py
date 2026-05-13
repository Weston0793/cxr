import io
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import streamlit as st
from PIL import Image

import pydicom



def resolve_cxas_class():
    import colorcet as cc
    from matplotlib.colors import ListedColormap

    def _lc_call(self, x, alpha=None, bytes=False):
        values = np.asarray(x)
        if np.issubdtype(values.dtype, np.integer):
            idx = np.mod(values, len(self.colors))
        else:
            mapped = np.clip(values, 0, 1)
            idx = np.minimum((mapped * (len(self.colors) - 1)).astype(int), len(self.colors) - 1)
        return np.asarray(self.colors)[idx]

    sample = ListedColormap([[0, 0, 0], [1, 1, 1]])
    ListedColormap.__call__ = _lc_call if not callable(sample) else ListedColormap.__call__

    cmap = getattr(cc.cm, "glasbey_bw_minc_20", None)
    if cmap is not None and not callable(cmap) and hasattr(cmap, "colors"):
        colors = np.asarray(cmap.colors)

        def _indexed_color(i: int):
            return tuple(colors[int(i) % len(colors)])

        cc.cm.glasbey_bw_minc_20 = _indexed_color

    import argparse
    import torch

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])

    if not getattr(torch.load, "_cxas_compat", False):
        _torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            map_location = kwargs.get("map_location")
            if not torch.cuda.is_available() and map_location == "cuda":
                kwargs["map_location"] = "cpu"
            return _torch_load(*args, **kwargs)

        _torch_load_compat._cxas_compat = True
        torch.load = _torch_load_compat

    import gdown

    _gdown_download = gdown.download

    def _normalize_drive_url(url: str) -> str:
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://drive.google.com/uc?id={match.group(1)}"
        return url

    def _download_compat(*args, **kwargs):
        fuzzy_requested = bool(kwargs.pop("fuzzy", False))
        if args:
            url = args[0]
            if fuzzy_requested and isinstance(url, str) and "drive.google.com" in url:
                args = (_normalize_drive_url(url),) + tuple(args[1:])
        elif fuzzy_requested and isinstance(kwargs.get("url"), str):
            kwargs["url"] = _normalize_drive_url(kwargs["url"])
        return _gdown_download(*args, **kwargs)

    gdown.download = _download_compat

    from cxas import CXAS

    return CXAS


st.set_page_config(page_title="CXR Anatomy Segmentation (CXAS)", layout="wide")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}
MAX_UPLOAD_MB = 25
MAX_INFERENCE_SIDE = 1536

GROUP_KEYWORDS: Dict[str, Sequence[str]] = {
    "airways_lungs": (
        "trachea",
        "tracheal bifurcation",
        "lung",
        "right lung",
        "left lung",
        "lung zone",
        "lobe",
    ),
    "cardiovascular": (
        "cardiomediastinum",
        "mediastinum",
        "heart",
        "atrium",
        "ventricle",
        "myocardium",
        "aorta",
        "pulmonary artery",
        "inferior vena cava",
    ),
    "skeletal": (
        "spine",
        "vertebra",
        "rib",
        "sternum",
        "rib cartilage",
        "clavicle",
        "scapula",
    ),
}

GROUP_COLORS = {
    "airways_lungs": np.array([46, 204, 113], dtype=np.uint8),
    "cardiovascular": np.array([231, 76, 60], dtype=np.uint8),
    "skeletal": np.array([52, 152, 219], dtype=np.uint8),
}


def infer_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@st.cache_resource(show_spinner=True)
def load_model(device: str):
    CXAS = resolve_cxas_class()

    def _build_model():
        init_attempts = [
            {"device": device},
            {"gpus": [device]},
            {"gpus": ["cpu"]},
            {},
        ]
        last_exc = None
        for kwargs in init_attempts:
            try:
                model_local = CXAS(**kwargs)
                if hasattr(model_local, "to"):
                    try:
                        model_local = model_local.to(device)
                    except Exception:
                        pass
                return model_local
            except TypeError as exc:
                last_exc = exc
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("Unable to initialize CXAS with known constructor signatures.")

    try:
        return _build_model()
    except Exception as exc:
        if "invalid load key" not in str(exc):
            raise
        weights_dir = Path.home() / ".cxas" / "weights"
        for f in weights_dir.glob("*.pth"):
            f.unlink(missing_ok=True)
        return _build_model()


def load_uploaded_image(uploaded_file) -> Tuple[np.ndarray, Path]:
    extension = uploaded_file.name.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {extension}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="cxas_upload_"))
    file_path = tmp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())

    if extension == "dcm":
        ds = pydicom.dcmread(str(file_path))
        pixel = ds.pixel_array.astype(np.float32)
        pixel -= pixel.min()
        if pixel.max() > 0:
            pixel /= pixel.max()
        image = (pixel * 255).astype(np.uint8)
    else:
        image = np.array(Image.open(io.BytesIO(uploaded_file.getvalue())).convert("L"))

    return image, file_path


def prepare_for_inference(image: np.ndarray, max_side: int = MAX_INFERENCE_SIDE) -> np.ndarray:
    """Bound image size to reduce OOM risk on Streamlit Community Cloud."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = Image.fromarray(image).resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _get_class_names(model) -> List[str]:
    candidates = ["class_names", "classes", "labels", "label_names"]
    for attr in candidates:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, (list, tuple)) and value:
                return [str(v) for v in value]
            if isinstance(value, dict) and value:
                return [str(v) for _, v in sorted(value.items())]
    raise RuntimeError("Could not discover class names from CXAS model object.")


def _run_inference(model, image_path: Path, image_gray: np.ndarray):
    input_path = str(image_path)
    attempted = []

    if hasattr(model, "forward"):
        try:
            import torch

            x = torch.from_numpy(image_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                return model(x)
        except Exception as exc:
            attempted.append(f"forward(tensor): {exc}")

    if hasattr(model, "process_file"):
        out_dir = Path(tempfile.mkdtemp(prefix="cxas_out_"))
        fn = getattr(model, "process_file")
        for args, kwargs in [
            ((input_path, str(out_dir)), {}),
            ((), {"input_path": input_path, "output_path": str(out_dir)}),
            ((), {"input_file": input_path, "output_dir": str(out_dir)}),
            ((input_path,), {}),
        ]:
            try:
                result = fn(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as exc:
                attempted.append(f"process_file args={args} kwargs={kwargs}: {exc}")

    def _try_call(fn, name: str):
        call_variants = [
            ((), {"image_path": input_path}),
            ((), {"file_path": input_path}),
            ((), {"input_path": input_path}),
            ((), {"path": input_path}),
            ((input_path,), {}),
            (([input_path],), {}),
        ]
        for args, kwargs in call_variants:
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                attempted.append(f"{name}{args or ''}{kwargs or ''}: {exc}")
                continue
        return None

    method_candidates = (
        "segment",
        "predict",
        "infer",
        "inference",
        "run",
        "run_inference",
        "predict_file",
        "process",
        "__call__",
    )

    for method in method_candidates:
        if hasattr(model, method):
            fn = getattr(model, method)
            result = _try_call(fn, method)
            if result is not None:
                return result

    available = [name for name in dir(model) if not name.startswith("_")]
    raise RuntimeError(
        "Could not find usable inference method on CXAS model. "
        f"Tried methods: {method_candidates}. Available attributes: {available[:40]}. "
        f"TypeErrors: {attempted[:6]}"
    )


def _extract_mask_tensor(prediction) -> np.ndarray:
    if isinstance(prediction, dict):
        for key in ("masks", "mask", "pred_masks", "segmentation", "logits"):
            if key in prediction:
                return np.asarray(prediction[key])
    if isinstance(prediction, (list, tuple)) and prediction:
        return np.asarray(prediction[0])
    return np.asarray(prediction)


def _to_class_first(masks: np.ndarray, n_classes: int) -> np.ndarray:
    if masks.ndim == 4 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim != 3:
        raise RuntimeError(f"Expected 3D mask tensor, got shape {masks.shape}.")
    if masks.shape[0] == n_classes:
        return masks
    if masks.shape[-1] == n_classes:
        return np.moveaxis(masks, -1, 0)
    raise RuntimeError(f"Could not align mask tensor shape {masks.shape} with {n_classes} classes.")


def class_indices_by_keywords(class_names: Iterable[str], keywords: Sequence[str]) -> List[int]:
    idx = []
    lower_names = [name.lower() for name in class_names]
    for i, class_name in enumerate(lower_names):
        if any(keyword in class_name for keyword in keywords):
            idx.append(i)
    return sorted(set(idx))


def build_group_masks(class_names: Sequence[str], masks_cls_first: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for group, keywords in GROUP_KEYWORDS.items():
        class_idxs = class_indices_by_keywords(class_names, keywords)
        if not class_idxs:
            out[group] = np.zeros(masks_cls_first.shape[1:], dtype=np.uint8)
            continue
        group_mask = (masks_cls_first[class_idxs].max(axis=0) > 0.5).astype(np.uint8)
        out[group] = group_mask
    return out


def overlay_mask(gray_image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    base = np.stack([gray_image] * 3, axis=-1).astype(np.float32)
    m = mask.astype(bool)
    base[m] = (1 - alpha) * base[m] + alpha * color
    return np.clip(base, 0, 255).astype(np.uint8)


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()


st.title("Chest X-ray Anatomy Segmentation (CXAS)")
st.caption("Inference-only Streamlit app powered by ConstantinSeibold/ChestXRayAnatomySegmentation via `cxas`.")

use_gpu = st.sidebar.checkbox("Use GPU (CUDA) if available", value=False)
device = infer_device(use_gpu)
if use_gpu and device == "cpu":
    st.sidebar.warning("CUDA unavailable in this environment. Falling back to CPU.")

model = None
model_load_error = None
try:
    model = load_model(device)
except Exception as exc:
    model_load_error = exc

if model_load_error is not None:
    st.error(
        "Model failed to load. If you are deploying on Streamlit Cloud, install `opencv-python-headless` "
        "(and avoid GUI OpenCV builds requiring `libGL.so.1`)."
    )
    st.exception(model_load_error)
    st.stop()

uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=sorted(ALLOWED_EXTENSIONS),
    max_upload_size=MAX_UPLOAD_MB,
    help=(
        "Large images can exceed Streamlit Community Cloud memory limits and restart the app. "
        f"Per-file upload is limited to {MAX_UPLOAD_MB} MB."
    ),
)

if uploaded is not None:
    try:
        with st.spinner("Loading image and running segmentation..."):
            image_gray, image_path = load_uploaded_image(uploaded)
            image_for_inference = prepare_for_inference(image_gray)
            if image_for_inference.shape != image_gray.shape:
                st.info(
                    f"Resized image from {image_gray.shape[1]}x{image_gray.shape[0]} to "
                    f"{image_for_inference.shape[1]}x{image_for_inference.shape[0]} for stable inference."
                )
            prediction = _run_inference(model, image_path, image_for_inference)
            class_names = _get_class_names(model)
            mask_tensor = _extract_mask_tensor(prediction)
            masks_cls_first = _to_class_first(mask_tensor, len(class_names))
            grouped_masks = build_group_masks(class_names, masks_cls_first)
    except Exception as exc:
        st.error("Segmentation failed. See details below.")
        st.exception(exc)
        st.info(
            "If the app stops without a Python traceback, it is usually an out-of-memory restart "
            "on Streamlit Community Cloud. Try a smaller image and keep GPU disabled."
        )
        st.stop()

    overlay_air = overlay_mask(image_gray, grouped_masks["airways_lungs"], GROUP_COLORS["airways_lungs"])
    overlay_card = overlay_mask(image_gray, grouped_masks["cardiovascular"], GROUP_COLORS["cardiovascular"])
    overlay_skel = overlay_mask(image_gray, grouped_masks["skeletal"], GROUP_COLORS["skeletal"])

    combined = np.stack([image_gray] * 3, axis=-1).astype(np.float32)
    for name, color in GROUP_COLORS.items():
        m = grouped_masks[name].astype(bool)
        combined[m] = 0.6 * combined[m] + 0.4 * color
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    col1, col2, col3, col4 = st.columns(4)
    col1.image(image_gray, caption="Original", use_container_width=True)
    col2.image(overlay_air, caption="Airways + lungs", use_container_width=True)
    col3.image(overlay_card, caption="Cardiovascular / mediastinal", use_container_width=True)
    col4.image(overlay_skel, caption="Skeletal", use_container_width=True)

    with st.expander("Optional combined overlay", expanded=False):
        st.image(combined, caption="Combined groups", use_container_width=True)

    st.subheader("Download group masks")
    download_specs = [
        ("Download airways + lungs mask", "airways_lungs", "airways_lungs_mask.png"),
        ("Download cardiovascular mask", "cardiovascular", "cardiovascular_mask.png"),
        ("Download skeletal mask", "skeletal", "skeletal_mask.png"),
    ]
    for col, (label, key, filename) in zip(st.columns(3), download_specs):
        col.download_button(
            label,
            data=mask_to_png_bytes(grouped_masks[key]),
            file_name=filename,
            mime="image/png",
        )
