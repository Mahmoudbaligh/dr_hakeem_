from __future__ import annotations

import gc
import traceback

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import CORSMiddleware

from .inference import (
    predict,
    explain,
    model_info,
    MAX_IMAGE_BYTES,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Dr. Hakeem AI API",
    description=(
        "AI-powered skin disease classification "
        "with ONNX inference and Grad-CAM explainability."
    ),
    version="3.1.0",
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
        "version": "3.1.0",

        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "explain": "/explain",
            "model_info": "/model-info",
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

        "onnx_loaded": info[
            "onnx_loaded"
        ],

        "gradcam_loaded": info[
            "gradcam_loaded"
        ],

        "device": info[
            "device"
        ],
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def get_model_info():

    return model_info()


# ============================================================
# READ IMAGE SAFELY
# ============================================================

async def read_image_file(
    file: UploadFile
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

    image_bytes = None

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Empty image."
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:

            raise HTTPException(
                status_code=413,
                detail=(
                    "Image is too large. "
                    "Maximum size is 8 MB."
                )
            )

        return image_bytes

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"[ERROR] Reading uploaded image failed: "
            f"{repr(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded image."
        )


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...)
):

    image_bytes = None

    try:

        # ----------------------------------------------------
        # Read image into memory only.
        # Nothing is saved to disk.
        # ----------------------------------------------------

        image_bytes = await read_image_file(
            file
        )

        # ----------------------------------------------------
        # AI prediction.
        # ----------------------------------------------------

        result = predict(
            image_bytes
        )

        return result

    except HTTPException:

        raise

    except ValueError as e:

        print(
            f"[WARN] Invalid prediction input: "
            f"{repr(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        print(
            "[ERROR] Prediction failed:"
        )

        print(
            repr(e)
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )

    finally:

        # ----------------------------------------------------
        # Close UploadFile.
        # ----------------------------------------------------

        try:

            await file.close()

        except Exception:

            pass

        # ----------------------------------------------------
        # Release image bytes.
        # ----------------------------------------------------

        image_bytes = None

        gc.collect()


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
        description=(
            "Heatmap overlay strength."
        )
    ),
):

    image_bytes = None

    try:

        # ----------------------------------------------------
        # Read image.
        # ----------------------------------------------------

        image_bytes = await read_image_file(
            file
        )

        # ----------------------------------------------------
        # Run Grad-CAM.
        # ----------------------------------------------------

        result = explain(
            image_bytes=image_bytes,
            target_class=target_class,
            alpha=alpha,
        )

        return result

    except HTTPException:

        raise

    except ValueError as e:

        print(
            "[WARN] Invalid Grad-CAM input:"
        )

        print(
            repr(e)
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        # ----------------------------------------------------
        # IMPORTANT:
        # Print the REAL error to Railway logs.
        # ----------------------------------------------------

        print(
            "=================================================="
        )

        print(
            "[ERROR] GRAD-CAM FAILED"
        )

        print(
            f"Exception type: {type(e).__name__}"
        )

        print(
            f"Exception: {repr(e)}"
        )

        traceback.print_exc()

        print(
            "=================================================="
        )

        # During development / debugging, returning the real
        # message makes it much easier to diagnose the issue.
        #
        # The full traceback remains server-side only.
        raise HTTPException(
            status_code=500,
            detail=(
                f"Grad-CAM failed: {str(e)}"
            )
        )

    finally:

        # ----------------------------------------------------
        # Close UploadFile.
        # ----------------------------------------------------

        try:

            await file.close()

        except Exception:

            pass

        # ----------------------------------------------------
        # Release image bytes.
        # ----------------------------------------------------

        image_bytes = None

        gc.collect()
