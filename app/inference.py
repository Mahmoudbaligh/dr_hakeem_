from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import onnxruntime as ort


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

ONNX_MODEL_PATH = MODEL_DIR / "model.onnx"
PYTORCH_MODEL_PATH = MODEL_DIR / "best_model_v4.pth"

CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 300

NUM_CLASSES = 5

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "nv",
    "mel",
]

CLASS_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
}


# These are the exact statistics from your training.
NORM_MEAN = [
    0.76264286,
    0.54455656,
    0.56845410,
]

NORM_STD = [
    0.14133665,
    0.15278324,
    0.17041880,
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD CLASS CONFIG IF AVAILABLE
# ============================================================

if CLASS_NAMES_PATH.exists():
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_config = json.load(f)

        CLASS_NAMES = class_config.get(
            "class_names",
            CLASS_NAMES
        )

        CLASS_LABELS = class_config.get(
            "class_labels",
            CLASS_LABELS
        )

    except Exception as e:
        print(f"[WARN] Could not read class_names.json: {e}")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORM_MEAN,
        std=NORM_STD
    ),
])


# ============================================================
# ONNX MODEL
# Used for normal /predict endpoint
# ============================================================

def create_onnx_session():

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {ONNX_MODEL_PATH}"
        )

    print(f"[INFO] Loading ONNX model: {ONNX_MODEL_PATH}")

    session_options = ort.SessionOptions()

    # Keep CPU usage reasonable on Railway
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    print(
        "[INFO] ONNX providers:",
        session.get_providers()
    )

    print(
        "[INFO] ONNX input:",
        session.get_inputs()[0].name,
        session.get_inputs()[0].shape
    )

    return session


SESSION = create_onnx_session()


# ============================================================
# PYTORCH MODEL
# Used ONLY for Grad-CAM
# ============================================================

def build_pytorch_model():

    print("[INFO] Building EfficientNet-B3 for Grad-CAM...")

    model = models.efficientnet_b3(
        weights=None
    )

    num_features = model.classifier[1].in_features

    # EXACT classifier structure used during training.
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.20),
        nn.Linear(
            num_features,
            NUM_CLASSES
        )
    )

    return model


def extract_state_dict(checkpoint: Any):

    """
    Handles common checkpoint formats:

    1. raw state_dict
    2. {"state_dict": ...}
    3. {"model_state_dict": ...}
    4. {"model": ...}
    """

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "model" in checkpoint:

            model_value = checkpoint["model"]

            if isinstance(model_value, dict):
                return model_value

    return checkpoint


def clean_state_dict(state_dict):

    """
    Removes prefixes such as:

    module.
    model.
    """

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_pytorch_model():

    if not PYTORCH_MODEL_PATH.exists():

        print(
            f"[WARN] PyTorch model not found: "
            f"{PYTORCH_MODEL_PATH}"
        )

        return None

    print(
        f"[INFO] Loading PyTorch Grad-CAM model: "
        f"{PYTORCH_MODEL_PATH}"
    )

    model = build_pytorch_model()

    checkpoint = torch.load(
        PYTORCH_MODEL_PATH,
        map_location="cpu"
    )

    state_dict = extract_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Unsupported checkpoint format. "
            "Expected a PyTorch state_dict."
        )

    state_dict = clean_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing:
        print(
            "[WARN] Missing keys:",
            len(missing)
        )

        print(
            missing[:10]
        )

    if unexpected:
        print(
            "[WARN] Unexpected keys:",
            len(unexpected)
        )

        print(
            unexpected[:10]
        )

    model = model.to(DEVICE)

    model.eval()

    print(
        f"[INFO] Grad-CAM PyTorch model loaded "
        f"on {DEVICE}"
    )

    return model


PYTORCH_MODEL = load_pytorch_model()


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

def get_gradcam_target_layer(model):

    """
    EfficientNet-B3 structure:

    model.features
        ...
        last feature block
            ...
                Conv2d

    We use the last Conv2d layer.
    """

    target_layer = None

    for module in model.modules():

        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is None:

        raise RuntimeError(
            "Could not find Conv2d layer "
            "for Grad-CAM."
        )

    return target_layer


# ============================================================
# GRAD-CAM ENGINE
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        output = self.model(input_tensor)

        if target_class is None:

            target_class = int(
                torch.argmax(output, dim=1).item()
            )

        score = output[:, target_class]

        score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations
        gradients = self.gradients

        # Global average pooling over H,W
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(cam)

        # Resize CAM to 300x300
        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return (
            output.detach(),
            cam.detach()
        )


# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_image(image_bytes: bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return image


def image_to_tensor(image: Image.Image):

    tensor = inference_transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# PREDICTION
# ============================================================

def softmax_numpy(logits):

    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(
        exp_logits,
        axis=1,
        keepdims=True
    )


def predict(
    image_bytes: bytes,
    top_k: int = 3
):

    image = load_image(
        image_bytes
    )

    tensor = image_to_tensor(
        image
    )

    input_array = (
        tensor.numpy()
        .astype(np.float32)
    )

    input_name = SESSION.get_inputs()[0].name

    outputs = SESSION.run(
        None,
        {
            input_name: input_array
        }
    )

    logits = np.asarray(
        outputs[0]
    )

    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    probabilities = softmax_numpy(
        logits
    )[0]

    predicted_idx = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_idx
    ]

    predicted_label = CLASS_LABELS.get(
        predicted_class,
        predicted_class
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    top_predictions = []

    for idx in top_indices:

        class_name = CLASS_NAMES[
            int(idx)
        ]

        top_predictions.append({
            "class": class_name,
            "label": CLASS_LABELS.get(
                class_name,
                class_name
            ),
            "confidence": float(
                probabilities[idx]
            )
        })

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": float(
            probabilities[predicted_idx]
        ),
        "top_predictions": top_predictions,
    }


# ============================================================
# GRAD-CAM IMAGE CREATION
# ============================================================

def create_heatmap_image(cam):

    """
    Creates a simple RGB heatmap without requiring OpenCV.
    """

    cam_uint8 = (
        cam * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # Blue -> Cyan -> Yellow -> Red style mapping
    r = cam_uint8

    g = np.clip(
        2 * cam_uint8,
        0,
        255
    ).astype(np.uint8)

    b = (
        255 - cam_uint8
    ).astype(np.uint8)

    heatmap = np.stack(
        [r, g, b],
        axis=2
    )

    return Image.fromarray(
        heatmap,
        mode="RGB"
    )


def create_overlay(
    original_image,
    heatmap_image,
    alpha=0.45
):

    original = original_image.resize(
        (IMG_SIZE, IMG_SIZE)
    ).convert("RGB")

    overlay = Image.blend(
        original,
        heatmap_image,
        alpha=float(alpha)
    )

    return overlay


def pil_to_base64(image: Image.Image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# EXPLAIN / GRAD-CAM
# ============================================================

def explain(
    image_bytes: bytes,
    target_class: int | None = None,
    alpha: float = 0.45
):

    if PYTORCH_MODEL is None:

        raise RuntimeError(
            "Grad-CAM model is not loaded. "
            "Make sure best_model_v4.pth exists."
        )

    original_image = load_image(
        image_bytes
    )

    input_tensor = image_to_tensor(
        original_image
    ).to(DEVICE)

    # We need gradients.
    input_tensor.requires_grad_(True)

    target_layer = get_gradcam_target_layer(
        PYTORCH_MODEL
    )

    gradcam = GradCAM(
        PYTORCH_MODEL,
        target_layer
    )

    try:

        logits, cam = gradcam.generate(
            input_tensor,
            target_class
        )

    finally:

        gradcam.remove_hooks()

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    predicted_idx = int(
        torch.argmax(
            probabilities
        ).item()
    )

    # If no explicit target class was supplied,
    # explain the predicted class.
    explained_idx = (
        predicted_idx
        if target_class is None
        else int(target_class)
    )

    if explained_idx < 0 or explained_idx >= NUM_CLASSES:

        raise ValueError(
            f"target_class must be between "
            f"0 and {NUM_CLASSES - 1}"
        )

    predicted_class = CLASS_NAMES[
        predicted_idx
    ]

    explained_class = CLASS_NAMES[
        explained_idx
    ]

    heatmap_image = create_heatmap_image(
        cam.cpu().numpy()
    )

    overlay_image = create_overlay(
        original_image,
        heatmap_image,
        alpha=alpha
    )

    top_indices = torch.argsort(
        probabilities,
        descending=True
    )[:3]

    top_predictions = []

    for idx in top_indices:

        idx = int(idx.item())

        class_name = CLASS_NAMES[idx]

        top_predictions.append({
            "class": class_name,
            "label": CLASS_LABELS.get(
                class_name,
                class_name
            ),
            "confidence": float(
                probabilities[idx].item()
            )
        })

    return {

        "predicted_class": predicted_class,

        "predicted_label": CLASS_LABELS.get(
            predicted_class,
            predicted_class
        ),

        "confidence": float(
            probabilities[predicted_idx].item()
        ),

        "explained_class": explained_class,

        "explained_label": CLASS_LABELS.get(
            explained_class,
            explained_class
        ),

        "explained_class_confidence": float(
            probabilities[explained_idx].item()
        ),

        "top_predictions": top_predictions,

        "heatmap_base64": pil_to_base64(
            heatmap_image
        ),

        "overlay_base64": pil_to_base64(
            overlay_image
        ),

        "image_size": {
            "width": IMG_SIZE,
            "height": IMG_SIZE
        },

        "gradcam": {
            "method": "Grad-CAM",
            "target_layer": "last Conv2d layer",
            "alpha": float(alpha)
        }
    }


# ============================================================
# MODEL INFO
# ============================================================

def model_info():

    return {

        "model": "EfficientNet-B3",

        "image_size": IMG_SIZE,

        "num_classes": NUM_CLASSES,

        "classes": CLASS_NAMES,

        "labels": CLASS_LABELS,

        "onnx_loaded": SESSION is not None,

        "gradcam_loaded": PYTORCH_MODEL is not None,

        "device": str(DEVICE),

        "gradcam_model": (
            "best_model_v4.pth"
            if PYTORCH_MODEL is not None
            else None
        )
    }