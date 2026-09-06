from __future__ import annotations

# ============================================================
# IMPORTANT:
# Set CPU-related env vars BEFORE importing torch.
# This helps reduce CPU thread/memory pressure on Railway.
# ============================================================

import os

os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import base64
import gc
import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
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

MAX_IMAGE_BYTES = 8 * 1024 * 1024       # 8 MB
MAX_IMAGE_PIXELS = 20_000_000            # 20 MP

# ------------------------------------------------------------
# OOD / rejection thresholds
# ------------------------------------------------------------
MIN_CONFIDENCE = 0.70
MIN_MARGIN = 0.15
MAX_ENTROPY = 0.85

# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------
DEFAULT_ALPHA = 0.45
MAX_ALPHA = 0.85
MIN_ALPHA = 0.10

# ------------------------------------------------------------
# Threading
# ------------------------------------------------------------
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ============================================================
# DEFAULT CLASSES
# ============================================================

DEFAULT_CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "nv",
    "mel",
]

DEFAULT_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
}


# ============================================================
# NORMALIZATION
# ============================================================

IMAGE_MEAN = [
    0.76264286,
    0.54455656,
    0.56845410,
]

IMAGE_STD = [
    0.14133665,
    0.15278324,
    0.17041880,
]


# ============================================================
# GLOBAL MODEL CONFIG
# ============================================================

CLASSES = DEFAULT_CLASSES.copy()
LABELS = DEFAULT_LABELS.copy()
NUM_CLASSES = len(CLASSES)


# ============================================================
# LOAD CLASS CONFIG
# ============================================================

def load_class_config() -> None:
    """
    Loads classes/labels/image size from class_names.json.
    Falls back to defaults when file is unavailable.
    """

    global CLASSES
    global LABELS
    global NUM_CLASSES

    if not CLASS_NAMES_PATH.exists():
        print("[WARN] class_names.json not found. Using defaults.")
        return

    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        classes = data.get("classes")
        labels = data.get("labels")
        img_size = data.get("img_size")
        num_classes = data.get("num_classes")

        if isinstance(classes, list) and classes:
            CLASSES = [str(x) for x in classes]

        if isinstance(labels, dict):
            LABELS = {
                str(k): str(v)
                for k, v in labels.items()
            }

        if isinstance(img_size, int) and img_size > 0:
            global IMG_SIZE
            IMG_SIZE = img_size

        NUM_CLASSES = len(CLASSES)

        if isinstance(num_classes, int) and num_classes != NUM_CLASSES:
            print(
                f"[WARN] class_names.json num_classes={num_classes}, "
                f"but classes={NUM_CLASSES}. Using classes length."
            )

        print(
            f"[INFO] Classes: {CLASSES} | "
            f"num_classes={NUM_CLASSES} | "
            f"image_size={IMG_SIZE}"
        )

    except Exception as e:
        print(f"[WARN] Failed to load class_names.json: {e}")


load_class_config()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

PYTORCH_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        ),
    ]
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[INFO] Grad-CAM device: {DEVICE}")


# ============================================================
# ONNX RUNTIME
# ============================================================

ONNX_SESSION: ort.InferenceSession | None = None
ONNX_INPUT_NAME: str | None = None


def load_onnx_model() -> None:
    """
    Loads the ONNX model once.
    ONNX is used for /predict because it is lighter and faster.
    """

    global ONNX_SESSION
    global ONNX_INPUT_NAME

    if ONNX_SESSION is not None:
        return

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {ONNX_MODEL_PATH}"
        )

    print(f"[INFO] Loading ONNX model: {ONNX_MODEL_PATH}")

    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    ONNX_SESSION = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    inputs = ONNX_SESSION.get_inputs()

    if not inputs:
        raise RuntimeError("ONNX model has no inputs.")

    ONNX_INPUT_NAME = inputs[0].name

    print(
        f"[INFO] ONNX providers: "
        f"{ONNX_SESSION.get_providers()}"
    )

    print(
        f"[INFO] ONNX input: "
        f"{ONNX_INPUT_NAME} "
        f"{inputs[0].shape}"
    )


load_onnx_model()


# ============================================================
# PYTORCH GRAD-CAM MODEL
# ============================================================

PYTORCH_MODEL: nn.Module | None = None

# Only one Grad-CAM request is processed at a time.
PYTORCH_MODEL_LOCK = threading.RLock()


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Supports:
      - raw state_dict
      - {'state_dict': ...}
      - {'model_state_dict': ...}
      - {'model': ...}
    """

    if isinstance(checkpoint, dict):

        for key in (
            "state_dict",
            "model_state_dict",
            "model",
        ):
            candidate = checkpoint.get(key)

            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported PyTorch checkpoint format."
        )

    cleaned: dict[str, torch.Tensor] = {}

    for key, value in checkpoint.items():

        if not isinstance(key, str):
            continue

        new_key = key

        # Remove DataParallel prefix.
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if isinstance(value, torch.Tensor):
            cleaned[new_key] = value

    if not cleaned:
        raise RuntimeError(
            "Checkpoint does not contain a valid state_dict."
        )

    return cleaned


def _checkpoint_uses_nested_classifier(
    state_dict: dict[str, torch.Tensor],
) -> bool:
    """
    Detects whether the trained checkpoint uses:

        classifier:
            Sequential(
                Dropout,
                Sequential(
                    Dropout,
                    Linear(...)
                )
            )

    or the normal torchvision form:

        classifier:
            Sequential(
                Dropout,
                Linear(...)
            )
    """

    nested_weight_keys = (
        "classifier.1.1.weight",
        "classifier.1.1.bias",
    )

    return any(
        key in state_dict
        for key in nested_weight_keys
    )


def build_pytorch_model(
    state_dict: dict[str, torch.Tensor],
) -> nn.Module:
    """
    Builds EfficientNet-B3 architecture matching the checkpoint.
    """

    print("[INFO] Building EfficientNet-B3 for Grad-CAM...")

    model = models.efficientnet_b3(
        weights=None
    )

    num_features = model.classifier[1].in_features

    if _checkpoint_uses_nested_classifier(state_dict):

        print(
            "[INFO] Using Sequential classifier architecture."
        )

        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.20),
            nn.Linear(
                num_features,
                NUM_CLASSES,
            ),
        )

    else:

        print(
            "[INFO] Using standard torchvision "
            "classifier architecture."
        )

        model.classifier[1] = nn.Linear(
            num_features,
            NUM_CLASSES,
        )

    return model


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================

def load_pytorch_model() -> nn.Module:
    """
    Loads Grad-CAM PyTorch model lazily.
    """

    global PYTORCH_MODEL

    if PYTORCH_MODEL is not None:
        return PYTORCH_MODEL

    if not PYTORCH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PyTorch checkpoint not found: "
            f"{PYTORCH_MODEL_PATH}"
        )

    print("[INFO] Loading PyTorch Grad-CAM model...")

    # Safe loading for a state-dict checkpoint.
    checkpoint = torch.load(
        str(PYTORCH_MODEL_PATH),
        map_location="cpu",
        weights_only=True,
    )

    state_dict = _extract_state_dict(checkpoint)

    print(
        "[INFO] Checkpoint classifier: "
        + (
            "Sequential"
            if _checkpoint_uses_nested_classifier(state_dict)
            else "Linear"
        )
    )

    model = build_pytorch_model(state_dict)

    # Strict loading is intentional.
    # We don't want to silently run Grad-CAM with random weights.
    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as e:

        raise RuntimeError(
            "PyTorch checkpoint does not exactly match "
            "the EfficientNet-B3 architecture.\n"
            f"{e}"
        ) from e

    model.eval()
    model.to(DEVICE)

    # We don't need parameter gradients.
    # Grad-CAM only needs gradients through activations.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    PYTORCH_MODEL = model

    print(
        f"[INFO] Grad-CAM model loaded successfully "
        f"on {DEVICE}"
    )

    return PYTORCH_MODEL


# ============================================================
# UNLOAD PYTORCH MODEL
# ============================================================

def unload_pytorch_model() -> None:
    """
    Releases PyTorch model and CPU memory.
    """

    global PYTORCH_MODEL

    PYTORCH_MODEL = None

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    image_bytes: bytes,
) -> Image.Image:
    """
    Validates and loads image without saving it to disk.
    """

    if not image_bytes:
        raise ValueError("Empty image.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            "Image is too large. Maximum size is 8 MB."
        )

    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            image.verify()

        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            rgb = image.convert("RGB")

            width, height = rgb.size

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Invalid image dimensions."
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Image has too many pixels."
                )

            # Copy so the underlying BytesIO/Image handle
            # is no longer required.
            rgb = rgb.copy()

        return rgb

    except ValueError:
        raise

    except Exception as e:
        raise ValueError(
            f"Invalid image: {e}"
        ) from e


# ============================================================
# NUMPY / PYTORCH PREPROCESSING
# ============================================================

def image_to_onnx_tensor(
    image: Image.Image,
) -> np.ndarray:
    """
    Converts image to ONNX input shape:
        [1, 3, H, W]
    """

    resized = image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    ) / 255.0

    mean = np.asarray(
        IMAGE_MEAN,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    std = np.asarray(
        IMAGE_STD,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    array = (array - mean) / std

    array = np.transpose(
        array,
        (2, 0, 1),
    )

    array = np.expand_dims(
        array,
        axis=0,
    )

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


def image_to_pytorch_tensor(
    image: Image.Image,
) -> torch.Tensor:
    """
    Converts image to PyTorch tensor:
        [1, 3, H, W]
    """

    tensor = PYTORCH_TRANSFORM(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        non_blocking=False,
    )

    # Required for backward/Grad-CAM while parameters stay frozen.
    tensor.requires_grad_(True)

    return tensor


# ============================================================
# PROBABILITY HELPERS
# ============================================================

def softmax_numpy(
    logits: np.ndarray,
) -> np.ndarray:
    """
    Numerically stable softmax.
    """

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp = np.exp(logits)

    denom = np.sum(
        exp,
        axis=1,
        keepdims=True,
    )

    return exp / np.clip(
        denom,
        1e-12,
        None,
    )


def calculate_metrics(
    probabilities: np.ndarray,
) -> tuple[int, float, float, float]:
    """
    Returns:
        best_index
        confidence
        margin
        normalized_entropy
    """

    probs = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probs.ndim != 1:
        raise ValueError(
            "Expected 1D probability vector."
        )

    sorted_probs = np.sort(
        probs
    )[::-1]

    best_index = int(
        np.argmax(probs)
    )

    confidence = float(
        sorted_probs[0]
    )

    second_probability = float(
        sorted_probs[1]
    ) if len(sorted_probs) > 1 else 0.0

    margin = confidence - second_probability

    entropy = -np.sum(
        probs * np.log(
            np.clip(
                probs,
                1e-12,
                1.0,
            )
        )
    )

    if len(probs) > 1:
        normalized_entropy = entropy / np.log(
            len(probs)
        )
    else:
        normalized_entropy = 0.0

    return (
        best_index,
        confidence,
        float(margin),
        float(normalized_entropy),
    )


def is_prediction_in_domain(
    confidence: float,
    margin: float,
    entropy: float,
) -> bool:
    """
    Confidence / margin / entropy based rejection.
    """

    return (
        confidence >= MIN_CONFIDENCE
        and margin >= MIN_MARGIN
        and entropy <= MAX_ENTROPY
    )


# ============================================================
# RESULT BUILDERS
# ============================================================

def build_success_result(
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """
    Creates the standard prediction response.
    """

    best_index, confidence, margin, entropy = (
        calculate_metrics(probabilities)
    )

    predicted_class = CLASSES[best_index]

    return {
        "status": "success",
        "is_in_domain": True,
        "diagnosis": LABELS.get(
            predicted_class,
            predicted_class,
        ),
        "class": predicted_class,
        "confidence": confidence,
    }


def build_rejection_result(
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """
    Creates object_not_defined response.
    """

    best_index, confidence, margin, entropy = (
        calculate_metrics(probabilities)
    )

    predicted_class = CLASSES[best_index]

    return {
        "status": "object_not_defined",
        "is_in_domain": False,
        "diagnosis": None,
        "class": None,
        "confidence": confidence,
        "message": (
            "The uploaded image could not be reliably "
            "identified as one of the supported skin-lesion classes."
        ),
        "rejection": {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
        },
    }


def result_from_probabilities(
    probabilities: np.ndarray,
) -> dict[str, Any]:

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if len(probabilities) != NUM_CLASSES:
        raise RuntimeError(
            f"Model returned {len(probabilities)} classes, "
            f"but API expects {NUM_CLASSES}."
        )

    (
        best_index,
        confidence,
        margin,
        entropy,
    ) = calculate_metrics(
        probabilities
    )

    if is_prediction_in_domain(
        confidence=confidence,
        margin=margin,
        entropy=entropy,
    ):
        return build_success_result(
            probabilities
        )

    return build_rejection_result(
        probabilities
    )


# ============================================================
# ONNX PREDICTION
# ============================================================

def predict(
    image_bytes: bytes,
) -> dict[str, Any]:
    """
    Standard production prediction endpoint.
    Uses ONNX only.
    """

    image = None
    input_tensor = None
    output = None

    try:

        image = load_image(
            image_bytes
        )

        input_tensor = image_to_onnx_tensor(
            image
        )

        if ONNX_SESSION is None:
            load_onnx_model()

        if ONNX_INPUT_NAME is None:
            raise RuntimeError(
                "ONNX input name is not available."
            )

        outputs = ONNX_SESSION.run(
            None,
            {
                ONNX_INPUT_NAME: input_tensor
            },
        )

        if not outputs:
            raise RuntimeError(
                "ONNX model returned no output."
            )

        output = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        if output.ndim == 1:
            output = output.reshape(
                1,
                -1,
            )

        if output.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                f"ONNX output has "
                f"{output.shape[1]} classes, "
                f"expected {NUM_CLASSES}."
            )

        probabilities = softmax_numpy(
            output
        )[0]

        return result_from_probabilities(
            probabilities
        )

    finally:

        del input_tensor
        del output
        image = None

        gc.collect()


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

def find_last_conv_layer(
    model: nn.Module,
) -> nn.Module:
    """
    Finds the last Conv2d layer.
    """

    last_conv = None

    for module in model.modules():

        if isinstance(
            module,
            nn.Conv2d,
        ):
            last_conv = module

    if last_conv is None:
        raise RuntimeError(
            "Could not find a Conv2d layer for Grad-CAM."
        )

    return last_conv


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """
    Generates normalized Grad-CAM heatmap in [0, 1].
    Output shape:
        [IMG_SIZE, IMG_SIZE]
    """

    target_layer = find_last_conv_layer(
        model
    )

    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def forward_hook(
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:

        nonlocal activations

        activations = output

    def backward_hook(
        module: nn.Module,
        grad_input: tuple[Any, ...],
        grad_output: tuple[Any, ...],
    ) -> None:

        nonlocal gradients

        if grad_output and grad_output[0] is not None:
            gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():

            logits = model(
                input_tensor
            )

            if logits.ndim != 2:
                raise RuntimeError(
                    f"Unexpected model output shape: "
                    f"{tuple(logits.shape)}"
                )

            if target_class < 0 or target_class >= logits.shape[1]:
                raise ValueError(
                    f"Invalid target class: {target_class}"
                )

            target_score = logits[
                0,
                target_class
            ]

            target_score.backward()

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # Global-average-pool gradients over spatial dimensions.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        cam = cam.detach().cpu().numpy()

        # Normalize [0,1]
        cam_min = float(
            np.min(cam)
        )

        cam_max = float(
            np.max(cam)
        )

        if cam_max - cam_min > 1e-8:
            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )
        else:
            cam = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        cam = np.clip(
            cam,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

        return cam

    finally:

        forward_handle.remove()
        backward_handle.remove()

        model.zero_grad(
            set_to_none=True
        )


# ============================================================
# HEATMAP COLORIZATION
# ============================================================

def create_heatmap_image(
    cam: np.ndarray,
) -> Image.Image:
    """
    Creates a color heatmap using PIL only.
    No OpenCV required.
    """

    cam_uint8 = np.clip(
        cam * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    # Simple blue -> cyan -> yellow -> red style.
    # Generated manually to avoid another heavy dependency.
    x = cam_uint8.astype(
        np.float32
    ) / 255.0

    r = np.clip(
        4.0 * x - 1.5,
        0.0,
        1.0,
    )

    g = np.clip(
        4.0 * np.minimum(x, 1.0 - x),
        0.0,
        1.0,
    )

    b = np.clip(
        1.5 - 4.0 * x,
        0.0,
        1.0,
    )

    rgb = np.stack(
        [
            r,
            g,
            b,
        ],
        axis=-1,
    )

    rgb = (
        rgb * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float,
) -> Image.Image:
    """
    Creates heatmap overlay at 300x300.
    """

    alpha = float(
        np.clip(
            alpha,
            MIN_ALPHA,
            MAX_ALPHA,
        )
    )

    base = image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR,
    ).convert(
        "RGB"
    )

    heatmap = create_heatmap_image(
        cam
    )

    return Image.blend(
        base,
        heatmap,
        alpha=alpha,
    )


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(
    image: Image.Image,
) -> str:
    """
    Encodes overlay as JPEG Base64.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=82,
        optimize=True,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode(
        "ascii"
    )


# ============================================================
# EXPLAIN
# ============================================================

def explain(
    image_bytes: bytes,
    target_class: int | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Production Grad-CAM pipeline.

    IMPORTANT:
    /explain DOES NOT call predict() first.
    It runs the PyTorch model exactly once, producing:
      - prediction
      - confidence
      - OOD/rejection decision
      - Grad-CAM
      - overlay

    This avoids loading ONNX + doing a second model inference
    during the same explain request.
    """

    image: Image.Image | None = None
    input_tensor: torch.Tensor | None = None
    model: nn.Module | None = None
    probabilities: np.ndarray | None = None
    cam: np.ndarray | None = None
    overlay: Image.Image | None = None

    with PYTORCH_MODEL_LOCK:

        try:

            print("[INFO] /explain started")

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = load_image(
                image_bytes
            )

            print(
                f"[INFO] Explain image: "
                f"{image.size[0]}x{image.size[1]}"
            )

            # ------------------------------------------------
            # Load PyTorch model
            # ------------------------------------------------

            model = load_pytorch_model()

            # ------------------------------------------------
            # Convert input
            # ------------------------------------------------

            input_tensor = image_to_pytorch_tensor(
                image
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            model.zero_grad(
                set_to_none=True
            )

            with torch.enable_grad():

                logits = model(
                    input_tensor
                )

            if logits.ndim != 2:
                raise RuntimeError(
                    f"Unexpected logits shape: "
                    f"{tuple(logits.shape)}"
                )

            if logits.shape[1] != NUM_CLASSES:
                raise RuntimeError(
                    f"PyTorch model returned "
                    f"{logits.shape[1]} classes, "
                    f"expected {NUM_CLASSES}."
                )

            probabilities_tensor = torch.softmax(
                logits,
                dim=1,
            )[0]

            probabilities = (
                probabilities_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            # ------------------------------------------------
            # Prediction metrics
            # ------------------------------------------------

            (
                best_index,
                confidence,
                margin,
                entropy,
            ) = calculate_metrics(
                probabilities
            )

            print(
                "[INFO] Explain prediction: "
                f"class={CLASSES[best_index]} | "
                f"confidence={confidence:.4f} | "
                f"margin={margin:.4f} | "
                f"entropy={entropy:.4f}"
            )

            # ------------------------------------------------
            # OOD / rejection
            # ------------------------------------------------

            in_domain = is_prediction_in_domain(
                confidence=confidence,
                margin=margin,
                entropy=entropy,
            )

            if not in_domain:

                print(
                    "[INFO] Explain request rejected "
                    "by OOD thresholds."
                )

                return build_rejection_result(
                    probabilities
                )

            # ------------------------------------------------
            # Determine Grad-CAM target
            # ------------------------------------------------

            if target_class is None:
                selected_class = best_index

            else:
                selected_class = int(
                    target_class
                )

                if (
                    selected_class < 0
                    or selected_class >= NUM_CLASSES
                ):
                    raise ValueError(
                        f"target_class must be between "
                        f"0 and {NUM_CLASSES - 1}."
                    )

            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            print(
                f"[INFO] Generating Grad-CAM "
                f"for class={selected_class}"
            )

            cam = generate_gradcam(
                model=model,
                input_tensor=input_tensor,
                target_class=selected_class,
            )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            overlay = create_overlay(
                image=image,
                cam=cam,
                alpha=alpha,
            )

            overlay_base64 = image_to_base64(
                overlay
            )

            predicted_class = CLASSES[
                best_index
            ]

            return {
                "status": "success",
                "is_in_domain": True,
                "diagnosis": LABELS.get(
                    predicted_class,
                    predicted_class,
                ),
                "class": predicted_class,
                "confidence": confidence,
                "overlay_base64": overlay_base64,
                "image_size": {
                    "width": IMG_SIZE,
                    "height": IMG_SIZE,
                },
                "gradcam": {
                    "method": "Grad-CAM",
                    "target_layer": "last Conv2d layer",
                    "alpha": float(
                        np.clip(
                            alpha,
                            MIN_ALPHA,
                            MAX_ALPHA,
                        )
                    ),
                    "target_class": selected_class,
                },
            }

        finally:

            print(
                "[INFO] Cleaning Grad-CAM resources..."
            )

            # Destroy local tensors/resources.
            try:
                if input_tensor is not None:
                    input_tensor.grad = None
            except Exception:
                pass

            input_tensor = None
            probabilities = None
            cam = None
            overlay = None
            image = None
            model = None

            # Release global model.
            unload_pytorch_model()

            gc.collect()

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            print(
                "[INFO] Grad-CAM resources released."
            )


# ============================================================
# MODEL INFO
# ============================================================

def get_model_info() -> dict[str, Any]:
    return {
        "model": "EfficientNet-B3",
        "framework": "PyTorch + ONNX Runtime",
        "image_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "device": str(DEVICE),
        "onnx_provider": (
            ONNX_SESSION.get_providers()
            if ONNX_SESSION is not None
            else []
        ),
        "gradcam": {
            "enabled": True,
            "method": "Grad-CAM",
            "target_layer": "last Conv2d layer",
        },
        "ood_rejection": {
            "enabled": True,
            "min_confidence": MIN_CONFIDENCE,
            "min_margin": MIN_MARGIN,
            "max_entropy": MAX_ENTROPY,
        },
    }


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "onnx_loaded": ONNX_SESSION is not None,
        "gradcam_loaded": PYTORCH_MODEL is not None,
        "device": str(DEVICE),
        "model": "EfficientNet-B3",
    }
