from __future__ import annotations

import base64
import gc
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import onnxruntime as ort


# ============================================================
# CPU / MEMORY SETTINGS
# ============================================================

# Railway normally runs CPU.
# This also prevents PyTorch from trying to use unsupported
# CPU optimizations such as NNPACK where possible.
os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")

if not torch.cuda.is_available():
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


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

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


# ============================================================
# OOD / REJECTION SETTINGS
# ============================================================
#
# These are NOT medical probabilities.
#
# They determine whether the classifier is confident enough
# to return a diagnosis.
#
# A prediction is rejected when:
#
#   confidence < MIN_CONFIDENCE
#       OR
#   margin < MIN_MARGIN
#       OR
#   entropy > MAX_ENTROPY
#
# These values should ideally be calibrated using validation
# and external non-skin images.
# ============================================================

MIN_CONFIDENCE = 0.70
MIN_MARGIN = 0.15
MAX_ENTROPY = 0.85


# ============================================================
# DEFAULT CLASSES
# ============================================================

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


# ============================================================
# NORMALIZATION
# ============================================================

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


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CLASS CONFIG
# ============================================================

if CLASS_NAMES_PATH.exists():

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            class_config = json.load(f)

        CLASS_NAMES = class_config.get(
            "class_names",
            CLASS_NAMES
        )

        CLASS_LABELS = class_config.get(
            "class_labels",
            CLASS_LABELS
        )

        IMG_SIZE = int(
            class_config.get(
                "img_size",
                IMG_SIZE
            )
        )

        NUM_CLASSES = int(
            class_config.get(
                "num_classes",
                NUM_CLASSES
            )
        )

    except Exception as e:

        print(
            f"[WARN] Could not read class_names.json: {e}"
        )


# ============================================================
# VALIDATE CLASS CONFIG
# ============================================================

if len(CLASS_NAMES) != NUM_CLASSES:

    raise RuntimeError(
        "CLASS_NAMES length does not match NUM_CLASSES. "
        f"Got {len(CLASS_NAMES)} classes but "
        f"NUM_CLASSES={NUM_CLASSES}."
    )


# ============================================================
# PIL SAFETY
# ============================================================

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# ============================================================
# IMAGE TRANSFORM
# ============================================================

inference_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORM_MEAN,
        std=NORM_STD
    ),
])


# ============================================================
# ONNX MODEL
# ============================================================

def create_onnx_session():

    if not ONNX_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"ONNX model not found: "
            f"{ONNX_MODEL_PATH}"
        )

    print(
        f"[INFO] Loading ONNX model: "
        f"{ONNX_MODEL_PATH}"
    )

    session_options = ort.SessionOptions()

    # Keep Railway CPU memory usage controlled.
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider"
        ],
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


# ============================================================
# LOAD ONNX ONCE
# ============================================================

SESSION = create_onnx_session()

ONNX_INPUT_NAME = (
    SESSION
    .get_inputs()[0]
    .name
)


# ============================================================
# PYTORCH GRAD-CAM MODEL
# ============================================================

PYTORCH_MODEL = None

# Protect model loading/unloading and Grad-CAM execution.
PYTORCH_MODEL_LOCK = threading.RLock()


# ============================================================
# BUILD EFFICIENTNET-B3
# ============================================================

def build_pytorch_model(
    sequential_classifier: bool = False
):

    print(
        "[INFO] Building EfficientNet-B3 "
        "for Grad-CAM..."
    )

    model = models.efficientnet_b3(
        weights=None
    )

    num_features = (
        model.classifier[1].in_features
    )

    # --------------------------------------------------------
    # Support BOTH possible checkpoint architectures.
    #
    # Architecture A:
    #
    # classifier[1] = Linear(...)
    #
    # Keys:
    # classifier.1.weight
    # classifier.1.bias
    #
    # Architecture B:
    #
    # classifier[1] = Sequential(
    #     Dropout,
    #     Linear
    # )
    #
    # Keys:
    # classifier.1.1.weight
    # classifier.1.1.bias
    # --------------------------------------------------------

    if sequential_classifier:

        model.classifier[1] = nn.Sequential(
            nn.Dropout(
                p=0.20
            ),
            nn.Linear(
                num_features,
                NUM_CLASSES
            )
        )

        print(
            "[INFO] Using Sequential classifier "
            "architecture."
        )

    else:

        model.classifier[1] = nn.Linear(
            num_features,
            NUM_CLASSES
        )

        print(
            "[INFO] Using standard Linear "
            "classifier architecture."
        )

    return model


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def extract_state_dict(
    checkpoint: Any
):

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            return checkpoint[
                "state_dict"
            ]

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]

        if "model" in checkpoint:

            model_value = checkpoint[
                "model"
            ]

            if isinstance(
                model_value,
                dict
            ):

                return model_value

    return checkpoint


def clean_state_dict(
    state_dict
):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module."):
            ]

        if new_key.startswith(
            "model."
        ):

            new_key = new_key[
                len("model."):]
        
        cleaned[new_key] = value

    return cleaned


def checkpoint_uses_sequential_classifier(
    state_dict
) -> bool:

    for key in state_dict.keys():

        if key.startswith(
            "classifier.1.1."
        ):

            return True

    return False


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================

def load_pytorch_model():

    if not PYTORCH_MODEL_PATH.exists():

        print(
            "[WARN] PyTorch Grad-CAM model "
            "not found:"
        )

        print(
            PYTORCH_MODEL_PATH
        )

        return None

    print(
        "[INFO] Loading PyTorch Grad-CAM model..."
    )

    # --------------------------------------------------------
    # Load checkpoint first so we can detect the exact
    # classifier architecture.
    # --------------------------------------------------------

    checkpoint = torch.load(
        PYTORCH_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    if not isinstance(
        state_dict,
        dict
    ):

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    state_dict = clean_state_dict(
        state_dict
    )

    sequential_classifier = (
        checkpoint_uses_sequential_classifier(
            state_dict
        )
    )

    print(
        "[INFO] Checkpoint classifier:",
        (
            "Sequential"
            if sequential_classifier
            else "Linear"
        )
    )

    # --------------------------------------------------------
    # Build matching architecture.
    # --------------------------------------------------------

    model = build_pytorch_model(
        sequential_classifier=
        sequential_classifier
    )

    # --------------------------------------------------------
    # Load weights.
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    # We do NOT silently accept an incompatible model.
    if missing:

        print(
            "[ERROR] Missing model keys:",
            len(missing)
        )

        print(
            "[ERROR] First missing keys:",
            missing[:20]
        )

        raise RuntimeError(
            "PyTorch checkpoint is incompatible "
            "with the EfficientNet-B3 architecture."
        )

    if unexpected:

        print(
            "[ERROR] Unexpected model keys:",
            len(unexpected)
        )

        print(
            "[ERROR] First unexpected keys:",
            unexpected[:20]
        )

        raise RuntimeError(
            "PyTorch checkpoint contains "
            "unexpected model parameters."
        )

    # --------------------------------------------------------
    # Move to device.
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"[INFO] Grad-CAM model loaded "
        f"successfully on {DEVICE}"
    )

    return model


# ============================================================
# GET / LOAD PYTORCH MODEL
# ============================================================

def get_or_load_pytorch_model():

    global PYTORCH_MODEL

    with PYTORCH_MODEL_LOCK:

        if PYTORCH_MODEL is not None:

            return PYTORCH_MODEL

        PYTORCH_MODEL = (
            load_pytorch_model()
        )

        return PYTORCH_MODEL


# ============================================================
# UNLOAD PYTORCH MODEL
# ============================================================

def unload_pytorch_model():

    global PYTORCH_MODEL

    with PYTORCH_MODEL_LOCK:

        if PYTORCH_MODEL is not None:

            try:

                PYTORCH_MODEL.zero_grad(
                    set_to_none=True
                )

            except Exception:

                pass

            try:

                PYTORCH_MODEL.cpu()

            except Exception:

                pass

            PYTORCH_MODEL = None

    gc.collect()

    if torch.cuda.is_available():

        try:

            torch.cuda.empty_cache()

        except Exception:

            pass


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

def get_gradcam_target_layer(
    model
):

    target_layer = None

    for module in model.modules():

        if isinstance(
            module,
            nn.Conv2d
        ):

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

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
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

        if grad_output:

            self.gradients = (
                grad_output[0]
            )

    def remove_hooks(self):

        if self.forward_handle:

            self.forward_handle.remove()

            self.forward_handle = None

        if self.backward_handle:

            self.backward_handle.remove()

            self.backward_handle = None

    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        self.activations = None
        self.gradients = None

        # Grad-CAM requires gradients.
        with torch.enable_grad():

            output = self.model(
                input_tensor
            )

            if target_class is None:

                target_class = int(
                    torch.argmax(
                        output,
                        dim=1
                    ).item()
                )

            score = output[
                0,
                target_class
            ]

            score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations "
                "were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients "
                "were not captured."
            )

        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )

        # Global average pooling of gradients.
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

        cam = torch.relu(
            cam
        )

        cam = torch.nn.functional.interpolate(
            cam,
            size=(
                IMG_SIZE,
                IMG_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze(
            0
        ).squeeze(
            0
        )

        cam_min = cam.min()
        cam_max = cam.max()

        denominator = (
            cam_max - cam_min + 1e-8
        )

        cam = (
            cam - cam_min
        ) / denominator

        # Detach everything returned from autograd.
        output_cpu = (
            output.detach()
            .cpu()
        )

        cam_cpu = (
            cam.detach()
            .cpu()
        )

        return (
            output_cpu,
            cam_cpu
        )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    image_bytes: bytes
):

    if not image_bytes:

        raise ValueError(
            "Empty image."
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:

        raise ValueError(
            "Image is too large. "
            "Maximum allowed size is 8 MB."
        )

    try:

        # First verify the actual image.
        with Image.open(
            io.BytesIO(image_bytes)
        ) as verify_image:

            verify_image.verify()

        # Re-open after verify because verify() invalidates
        # the image object.
        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            image = image.convert(
                "RGB"
            )

            # Copy so the BytesIO / PIL object can be released.
            image = image.copy()

    except Exception as e:

        raise ValueError(
            f"Invalid image: {str(e)}"
        )

    return image


# ============================================================
# IMAGE -> TENSOR
# ============================================================

def image_to_tensor(
    image: Image.Image
):

    tensor = inference_transform(
        image
    )

    return tensor.unsqueeze(
        0
    )


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def softmax_numpy(
    logits
):

    logits = (
        logits -
        np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_logits = np.exp(
        logits
    )

    probabilities = (
        exp_logits /
        np.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )
    )

    return probabilities


def calculate_entropy(
    probabilities
):

    probabilities = np.clip(
        probabilities,
        1e-10,
        1.0
    )

    entropy = -np.sum(
        probabilities *
        np.log(
            probabilities
        )
    )

    max_entropy = np.log(
        len(probabilities)
    )

    if max_entropy <= 0:

        return 0.0

    return float(
        entropy / max_entropy
    )


def evaluate_prediction(
    probabilities
):

    sorted_probs = np.sort(
        probabilities
    )[::-1]

    confidence = float(
        sorted_probs[0]
    )

    second_confidence = float(
        sorted_probs[1]
    )

    margin = (
        confidence -
        second_confidence
    )

    entropy = calculate_entropy(
        probabilities
    )

    accepted = (
        confidence >= MIN_CONFIDENCE
        and margin >= MIN_MARGIN
        and entropy <= MAX_ENTROPY
    )

    return {
        "accepted": bool(
            accepted
        ),
        "confidence": confidence,
        "margin": float(
            margin
        ),
        "entropy": float(
            entropy
        ),
    }


# ============================================================
# PREDICTION
# ============================================================

def predict(
    image_bytes: bytes
):

    image = None
    tensor = None
    input_array = None
    outputs = None
    logits = None
    probabilities = None

    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = load_image(
            image_bytes
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        tensor = image_to_tensor(
            image
        )

        input_array = (
            tensor.numpy()
            .astype(
                np.float32,
                copy=False
            )
        )

        # ----------------------------------------------------
        # ONNX inference
        # ----------------------------------------------------

        outputs = SESSION.run(
            None,
            {
                ONNX_INPUT_NAME:
                input_array
            }
        )

        if not outputs:

            raise RuntimeError(
                "ONNX model returned no output."
            )

        logits = np.asarray(
            outputs[0]
        )

        if logits.ndim == 1:

            logits = logits.reshape(
                1,
                -1
            )

        if logits.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                "Model output class count "
                f"({logits.shape[1]}) does not match "
                f"configured NUM_CLASSES "
                f"({NUM_CLASSES})."
            )

        probabilities = (
            softmax_numpy(
                logits
            )[0]
        )

        predicted_idx = int(
            np.argmax(
                probabilities
            )
        )

        quality = evaluate_prediction(
            probabilities
        )

        # ----------------------------------------------------
        # REJECT UNKNOWN / LOW-CONFIDENCE
        # ----------------------------------------------------

        if not quality["accepted"]:

            return {
                "status": "object_not_defined",
                "is_in_domain": False,
                "diagnosis": None,
                "class": None,

                # This is the ACTUAL maximum model confidence.
                # It is not fabricated.
                "confidence": quality[
                    "confidence"
                ],

                "message": (
                    "The uploaded image is outside "
                    "the supported domain or the "
                    "model confidence is insufficient "
                    "for a reliable prediction."
                ),

                "rejection": {
                    "confidence": quality[
                        "confidence"
                    ],
                    "margin": quality[
                        "margin"
                    ],
                    "entropy": quality[
                        "entropy"
                    ],
                }
            }

        # ----------------------------------------------------
        # ACCEPTED PREDICTION
        # ----------------------------------------------------

        if predicted_idx >= len(
            CLASS_NAMES
        ):

            raise RuntimeError(
                "Predicted class index is "
                "outside CLASS_NAMES."
            )

        predicted_class = (
            CLASS_NAMES[
                predicted_idx
            ]
        )

        predicted_label = (
            CLASS_LABELS.get(
                predicted_class,
                predicted_class
            )
        )

        return {
            "status": "success",
            "is_in_domain": True,

            # One diagnosis only.
            "diagnosis": predicted_label,

            "class": predicted_class,

            # Actual probability from model.
            "confidence": quality[
                "confidence"
            ]
        }

    finally:

        # ----------------------------------------------------
        # MEMORY CLEANUP
        # ----------------------------------------------------

        image = None
        tensor = None
        input_array = None
        outputs = None
        logits = None
        probabilities = None

        gc.collect()


# ============================================================
# HEATMAP IMAGE
# ============================================================

def create_heatmap_image(
    cam
):

    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    cam = np.clip(
        cam,
        0.0,
        1.0
    )

    cam_uint8 = (
        cam * 255
    ).astype(
        np.uint8
    )

    # Simple RGB heatmap.
    r = cam_uint8

    g = np.clip(
        2 * cam_uint8,
        0,
        255
    ).astype(
        np.uint8
    )

    b = (
        255 -
        cam_uint8
    ).astype(
        np.uint8
    )

    heatmap = np.stack(
        [
            r,
            g,
            b
        ],
        axis=2
    )

    return Image.fromarray(
        heatmap,
        mode="RGB"
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap_image,
    alpha=0.45
):

    original = (
        original_image
        .resize(
            (
                IMG_SIZE,
                IMG_SIZE
            ),
            Image.Resampling.BILINEAR
        )
        .convert(
            "RGB"
        )
    )

    return Image.blend(
        original,
        heatmap_image,
        alpha=float(alpha)
    )


# ============================================================
# IMAGE -> BASE64
# ============================================================

def pil_to_base64(
    image: Image.Image
):

    buffer = io.BytesIO()

    try:

        image.save(
            buffer,
            format="JPEG",
            quality=80,
            optimize=True
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode(
            "utf-8"
        )

        return encoded

    finally:

        buffer.close()


# ============================================================
# EXPLAIN / GRAD-CAM
# ============================================================

def explain(
    image_bytes: bytes,
    target_class: int | None = None,
    alpha: float = 0.45
):

    # --------------------------------------------------------
    # FIRST:
    # Run ONNX prediction.
    #
    # If rejected, we NEVER load the PyTorch model.
    # --------------------------------------------------------

    prediction = predict(
        image_bytes
    )

    if prediction["status"] != "success":

        return prediction

    original_image = None
    input_tensor = None
    gradcam = None
    target_layer = None
    logits = None
    cam = None
    heatmap_image = None
    overlay_image = None
    overlay_base64 = None

    # --------------------------------------------------------
    # Lock the whole Grad-CAM operation.
    #
    # This prevents two requests from:
    #
    # Request A -> loading model
    # Request B -> unloading model
    #
    # at the same time.
    # --------------------------------------------------------

    with PYTORCH_MODEL_LOCK:

        model = None

        try:

            # ------------------------------------------------
            # Load PyTorch model ONLY for explain.
            # ------------------------------------------------

            model = get_or_load_pytorch_model()

            if model is None:

                raise RuntimeError(
                    "Grad-CAM model is unavailable."
                )

            # ------------------------------------------------
            # Load original image.
            # ------------------------------------------------

            original_image = load_image(
                image_bytes
            )

            # ------------------------------------------------
            # Convert to tensor.
            # ------------------------------------------------

            input_tensor = (
                image_to_tensor(
                    original_image
                )
                .to(
                    DEVICE
                )
            )

            input_tensor.requires_grad_(
                True
            )

            # ------------------------------------------------
            # Determine class.
            # ------------------------------------------------

            predicted_idx = (
                CLASS_NAMES.index(
                    prediction["class"]
                )
            )

            if target_class is None:

                explained_idx = (
                    predicted_idx
                )

            else:

                explained_idx = int(
                    target_class
                )

            if (
                explained_idx < 0
                or explained_idx >= NUM_CLASSES
            ):

                raise ValueError(
                    f"target_class must be "
                    f"between 0 and "
                    f"{NUM_CLASSES - 1}"
                )

            # ------------------------------------------------
            # Target layer.
            # ------------------------------------------------

            target_layer = (
                get_gradcam_target_layer(
                    model
                )
            )

            # ------------------------------------------------
            # Grad-CAM.
            # ------------------------------------------------

            gradcam = GradCAM(
                model,
                target_layer
            )

            try:

                logits, cam = (
                    gradcam.generate(
                        input_tensor,
                        explained_idx
                    )
                )

            finally:

                gradcam.remove_hooks()

            # ------------------------------------------------
            # Create heatmap.
            # ------------------------------------------------

            heatmap_image = (
                create_heatmap_image(
                    cam.numpy()
                )
            )

            # ------------------------------------------------
            # Create overlay.
            # ------------------------------------------------

            overlay_image = (
                create_overlay(
                    original_image,
                    heatmap_image,
                    alpha=alpha
                )
            )

            # ------------------------------------------------
            # Convert overlay to Base64.
            # ------------------------------------------------

            overlay_base64 = (
                pil_to_base64(
                    overlay_image
                )
            )

            # ------------------------------------------------
            # FINAL RESPONSE
            # ------------------------------------------------

            return {
                "status": "success",
                "is_in_domain": True,

                "diagnosis": prediction[
                    "diagnosis"
                ],

                "class": prediction[
                    "class"
                ],

                "confidence": prediction[
                    "confidence"
                ],

                "overlay_base64": (
                    overlay_base64
                ),

                "image_size": {
                    "width": IMG_SIZE,
                    "height": IMG_SIZE
                },

                "gradcam": {
                    "method": "Grad-CAM",
                    "target_layer": (
                        "last Conv2d layer"
                    ),
                    "alpha": float(
                        alpha
                    )
                }
            }

        finally:

            # ------------------------------------------------
            # Remove hooks.
            # ------------------------------------------------

            if gradcam is not None:

                try:

                    gradcam.remove_hooks()

                except Exception:

                    pass

            # ------------------------------------------------
            # Clear PyTorch gradients.
            # ------------------------------------------------

            if model is not None:

                try:

                    model.zero_grad(
                        set_to_none=True
                    )

                except Exception:

                    pass

            # ------------------------------------------------
            # Release large objects.
            # ------------------------------------------------

            original_image = None
            input_tensor = None
            gradcam = None
            target_layer = None
            logits = None
            cam = None
            heatmap_image = None
            overlay_image = None
            overlay_base64 = None

            # ------------------------------------------------
            # IMPORTANT:
            # Unload PyTorch after EVERY explain request.
            # ONNX remains loaded.
            # ------------------------------------------------

            unload_pytorch_model()

            gc.collect()


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

        "onnx_loaded": (
            SESSION is not None
        ),

        "gradcam_loaded": (
            PYTORCH_MODEL is not None
        ),

        "device": str(
            DEVICE
        ),

        "ood": {

            "min_confidence":
                MIN_CONFIDENCE,

            "min_margin":
                MIN_MARGIN,

            "max_entropy":
                MAX_ENTROPY
        },

        "memory_policy": {

            "onnx_persistent": True,

            "pytorch_gradcam_persistent":
                False,

            "images_saved_to_server":
                False,

            "max_image_bytes":
                MAX_IMAGE_BYTES
        },

        "gradcam_model": (
            "best_model_v4.pth"
            if PYTORCH_MODEL is not None
            else None
        )
    }
