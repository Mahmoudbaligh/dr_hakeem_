from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path

# Ensure root directory is in sys.path when running main.py directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from .inference import (
        predict,
        explain,
        model_info,
        load_image,
        pil_to_base64,
        IMG_SIZE,
    )
    from .gemini_service import check_image_with_gemini
except ImportError:
    from app.inference import (
        predict,
        explain,
        model_info,
        load_image,
        pil_to_base64,
        IMG_SIZE,
    )
    from app.gemini_service import check_image_with_gemini


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Dr. Hakeem AI API",
    description=(
        "AI-powered skin disease classification "
        "with ONNX inference and Grad-CAM explainability."
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Dr. Hakeem AI API",
        "status": "online",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "explain": "/explain",
            "docs": "/docs",
        }
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    info = model_info()

    return {
        "status": "healthy",
        "onnx_loaded": info["onnx_loaded"],
        "gradcam_loaded": info["gradcam_loaded"],
        "device": info["device"],
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def get_model_info():

    return model_info()


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...)
):

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type is missing."
        )

    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Empty image."
            )

        gemini_result = check_image_with_gemini(image_bytes)
        
        if gemini_result == "NOT_SKIN":
            return {
                "success": True,
                "filename": file.filename,
                "predicted_class": "not_skin",
                "predicted_label": "Not a skin image",
                "confidence": 1.0,
                "top_predictions": [
                    {
                        "class": "not_skin",
                        "label": "Not a skin image",
                        "confidence": 1.0
                    }
                ],
                "status": "not_skin",
                "message": "The image is not a skin image."
            }
            
        elif gemini_result == "HEALTHY":
            return {
                "success": True,
                "filename": file.filename,
                "predicted_class": "healthy",
                "predicted_label": "Healthy skin",
                "confidence": 1.0,
                "top_predictions": [
                    {
                        "class": "healthy",
                        "label": "Healthy skin",
                        "confidence": 1.0
                    }
                ],
                "status": "healthy",
                "message": "The skin appears healthy."
            }
            
        # Fallback to the custom model if it's a disease or error
        result = predict(
            image_bytes
        )

        return {
            "success": True,
            "filename": file.filename,
            "status": "disease",
            **result
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# EXPLAIN / GRAD-CAM
# ============================================================

@app.post("/explain")
async def explain_endpoint(

    file: UploadFile = File(...),

    target_class: int | None = Query(
        default=None,
        ge=0,
        le=4,
        description=(
            "Optional class index to explain. "
            "If omitted, the predicted class is explained."
        )
    ),

    alpha: float = Query(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Heatmap overlay strength."
    ),
):

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type is missing."
        )

    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Empty image."
            )

        gemini_result = check_image_with_gemini(image_bytes)
        
        if gemini_result == "NOT_SKIN":
            raw_img = load_image(image_bytes)
            img_b64 = pil_to_base64(raw_img)
            return {
                "success": True,
                "filename": file.filename,
                "predicted_class": "not_skin",
                "predicted_label": "Not a skin image",
                "confidence": 1.0,
                "explained_class": "not_skin",
                "explained_label": "Not a skin image",
                "explained_class_confidence": 1.0,
                "top_predictions": [
                    {
                        "class": "not_skin",
                        "label": "Not a skin image",
                        "confidence": 1.0
                    }
                ],
                "heatmap_base64": img_b64,
                "overlay_base64": img_b64,
                "image_size": {
                    "width": IMG_SIZE,
                    "height": IMG_SIZE
                },
                "gradcam": {
                    "method": "Grad-CAM",
                    "target_layer": "N/A",
                    "alpha": float(alpha)
                },
                "status": "not_skin",
                "message": "The image is not a skin image."
            }
            
        elif gemini_result == "HEALTHY":
            raw_img = load_image(image_bytes)
            img_b64 = pil_to_base64(raw_img)
            return {
                "success": True,
                "filename": file.filename,
                "predicted_class": "healthy",
                "predicted_label": "Healthy skin",
                "confidence": 1.0,
                "explained_class": "healthy",
                "explained_label": "Healthy skin",
                "explained_class_confidence": 1.0,
                "top_predictions": [
                    {
                        "class": "healthy",
                        "label": "Healthy skin",
                        "confidence": 1.0
                    }
                ],
                "heatmap_base64": img_b64,
                "overlay_base64": img_b64,
                "image_size": {
                    "width": IMG_SIZE,
                    "height": IMG_SIZE
                },
                "gradcam": {
                    "method": "Grad-CAM",
                    "target_layer": "N/A",
                    "alpha": float(alpha)
                },
                "status": "healthy",
                "message": "The skin appears healthy."
            }

        result = explain(
            image_bytes=image_bytes,
            target_class=target_class,
            alpha=alpha,
        )

        return {
            "success": True,
            "filename": file.filename,
            "status": "disease",
            **result
        }

    except HTTPException:
        raise

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)