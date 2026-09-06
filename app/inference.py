from __future__ import annotations

# ============================================================
# CPU / THREAD SETTINGS
# Must be set before importing torch.
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

try:
    torch.backends.nnpack.enabled = False
except Exception:
    pass

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
# Grad-CAM overlay
# ------------------------------------------------------------

DEFAULT_ALPHA = 0.45
MAX_ALPHA = 0.85
MIN_ALPHA = 0.10

# ------------------------------------------------------------
# CPU threading
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
    global IMG_SIZE

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
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
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
    Loads the ONNX model lazily.
    ONNX is used only for /predict.
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


def unload_onnx_model() -> None:
    """
    Releases ONNX Runtime session.
    This prevents ONNX + PyTorch from occupying RAM together
    during /explain.
    """

    global ONNX_SESSION
    global ONNX_INPUT_NAME

    ONNX_SESSION = None
    ONNX_INPUT_NAME = None

    gc.collect()


# ============================================================
# PYTORCH GRAD-CAM MODEL
# ============================================================

PYTORCH_MODEL: nn.Module | None = None

# Only one Grad-CAM request at a time.
PYTORCH_MODEL_LOCK = threading.RLock()


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
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
    Detects whether the trained checkpoint uses a nested
    classifier:

        classifier.1:
            Sequential(
                Dropout,
                Linear(...)
            )

    versus the normal torchvision form:

        classifier.1:
            Linear(...)
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
    # Never run Grad-CAM with random / partially loaded weights.
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

    # Grad-CAM implementation below does not require parameter
    # gradients. We use a detached activation map instead.
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
    Releases PyTorch model and memory.
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
    Validates and loads the image.

    IMPORTANT:
    The original phone image can be 3072x3072 or larger.
    We resize it immediately to IMG_SIZE x IMG_SIZE to
    avoid carrying a huge PIL image through /explain.
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

            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Invalid image dimensions."
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Image has too many pixels."
                )

            rgb = image.convert("RGB")

            # Resize immediately to model resolution.
            rgb = rgb.resize(
                (IMG_SIZE, IMG_SIZE),
                Image.Resampling.BILINEAR,
            )

            return rgb.copy()

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

    No requires_grad is needed because the memory-efficient
    Grad-CAM implementation uses a forward activation map only.
    """

    tensor = PYTORCH_TRANSFORM(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        non_blocking=False,
    )

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

    second_probability = (
        float(sorted_probs[1])
        if len(sorted_probs) > 1
        else 0.0
    )

    margin = (
        confidence
        - second_probability
    )

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
        normalized_entropy = (
            entropy
            / np.log(len(probs))
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

    (
        best_index,
        confidence,
        _margin,
        _entropy,
    ) = calculate_metrics(
        probabilities
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
    }


def build_rejection_result(
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """
    Creates object_not_defined response.
    """

    (
        best_index,
        confidence,
        margin,
        entropy,
    ) = calculate_metrics(
        probabilities
    )

    predicted_class = CLASSES[
        best_index
    ]

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
        _best_index,
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

    image: Image.Image | None = None
    input_tensor: np.ndarray | None = None
    output: np.ndarray | None = None

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

        input_tensor = None
        output = None
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

    last_conv: nn.Module | None = None

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
# MEMORY-EFFICIENT FINAL-LAYER GRAD-CAM
# ============================================================

def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """
    Memory-efficient final-layer Grad-CAM / CAM.

    EfficientNet-B3 ends approximately as:

        final convolution feature maps
                ↓
        global average pooling
                ↓
        classifier

    Because the last convolution feeds a GAP + Linear classifier,
    the class activation weighting can be computed directly from
    the classifier weights.

    This avoids a full autograd backward graph and dramatically
    lowers CPU/RAM usage on Railway.

    Returns:
        heatmap in [0, 1] with shape [IMG_SIZE, IMG_SIZE]
    """

    target_layer = find_last_conv_layer(
        model
    )

    captured: dict[str, torch.Tensor] = {}

    def forward_hook(
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:

        # Detach immediately so the computation graph is not kept.
        captured["activations"] = output.detach()

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        # ----------------------------------------------------
        # Forward only: no autograd graph
        # ----------------------------------------------------

        with torch.inference_mode():

            logits = model(
                input_tensor
            )

        if logits.ndim != 2:
            raise RuntimeError(
                f"Unexpected model output shape: "
                f"{tuple(logits.shape)}"
            )

        if (
            target_class < 0
            or target_class >= logits.shape[1]
        ):
            raise ValueError(
                f"Invalid target class: {target_class}"
            )

        if "activations" not in captured:
            raise RuntimeError(
                "Could not capture final convolution activations."
            )

        activations = captured[
            "activations"
        ]

        # ----------------------------------------------------
        # Find final Linear classifier
        # ----------------------------------------------------

        classifier_layer = model.classifier[1]

        if isinstance(
            classifier_layer,
            nn.Sequential,
        ):

            linear_layer: nn.Linear | None = None

            for module in classifier_layer.modules():

                if isinstance(
                    module,
                    nn.Linear,
                ):
                    linear_layer = module

            if linear_layer is None:
                raise RuntimeError(
                    "Could not find Linear layer "
                    "inside classifier."
                )

        elif isinstance(
            classifier_layer,
            nn.Linear,
        ):

            linear_layer = classifier_layer

        else:

            raise RuntimeError(
                "Unsupported EfficientNet classifier architecture."
            )

        # ----------------------------------------------------
        # Direct CAM weighting from classifier
        # ----------------------------------------------------

        class_weights = linear_layer.weight[
            target_class
        ].detach()

        # activations:   [1, C, H, W]
        # class_weights: [C]
        cam = (
            activations[0]
            * class_weights[:, None, None]
        ).sum(
            dim=0
        )

        # Positive evidence only.
        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize heatmap
        # ----------------------------------------------------

        cam = cam.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        cam = F.interpolate(
            cam,
            size=(
                IMG_SIZE,
                IMG_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        cam = cam.detach().cpu().numpy()

        # ----------------------------------------------------
        # Normalize to [0, 1]
        # ----------------------------------------------------

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )

        if (
            cam_max - cam_min
            > 1e-8
        ):

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

        return np.clip(
            cam,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    finally:

        handle.remove()

        captured.clear()

        try:
            model.zero_grad(
                set_to_none=True
            )
        except Exception:
            pass

        gc.collect()


# ============================================================
# HEATMAP COLORIZATION
# ============================================================

def create_heatmap_image(
    cam: np.ndarray,
) -> Image.Image:
    """
    Creates a heatmap using PIL/NumPy only.
    No OpenCV required.
    """

    x = np.clip(
        cam,
        0.0,
        1.0,
    ).astype(
        np.float32
    )

    # Blue -> Cyan -> Yellow -> Red.
    r = np.clip(
        4.0 * x - 1.5,
        0.0,
        1.0,
    )

    g = np.clip(
        4.0 * np.minimum(
            x,
            1.0 - x,
        ),
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
    Creates heatmap overlay at model resolution.
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
# EXPLAIN / GRAD-CAM
# ============================================================

def explain(
    image_bytes: bytes,
    target_class: int | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Production Grad-CAM pipeline.

    IMPORTANT:
    - /explain never calls /predict.
    - ONNX memory is released first.
    - PyTorch performs one forward pass only.
    - No backward/autograd graph is built.
    - Heatmap is computed from final-conv activations + classifier
      weights.
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
            # IMPORTANT:
            # Free ONNX before loading PyTorch.
            # This avoids holding both models in RAM.
            # ------------------------------------------------

            unload_onnx_model()
            gc.collect()

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
            # Prediction
            # ------------------------------------------------

            with torch.inference_mode():

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
            # Free tiny forward tensors before CAM.
            # Model stays loaded, input stays loaded.
            # ------------------------------------------------

            logits = None
            probabilities_tensor = None

            gc.collect()

            # ------------------------------------------------
            # Memory-efficient Grad-CAM
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

            # Release local tensors/resources.
            input_tensor = None
            probabilities = None
            cam = None
            overlay = None
            image = None
            model = None

            # Release global model.
            unload_pytorch_model()

            # Make sure ONNX is also not left around.
            unload_onnx_model()

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
            "method": "Memory-efficient final-layer Grad-CAM",
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
