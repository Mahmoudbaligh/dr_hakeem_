from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.inference import (
    MAX_IMAGE_BYTES,
    explain,
    get_model_info,
    health_check,
    predict,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "dr-hakeem-api"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Dr. Hakeem AI API",
    description=(
        "AI-powered skin lesion classification "
        "and Grad-CAM explainability API."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

async def read_image_file(
    file: UploadFile,
) -> bytes:
    """
    Reads uploaded image into memory.

    The API DOES NOT save the uploaded image to disk.
    """

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Image filename is missing.",
            )

        # Normal Flutter multipart uploads should use image/*
        # but we don't reject unknown content-types here because
        # some clients send application/octet-stream.
        content_type = (
            file.content_type or ""
        ).lower()

        logger.info(
            "Incoming file: name=%s content_type=%s",
            file.filename,
            content_type,
        )

        content = await file.read(
            MAX_IMAGE_BYTES + 1
        )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Image is too large. "
                    "Maximum size is 8 MB."
                ),
            )

        return content

    finally:

        try:
            await file.close()
        except Exception:
            pass


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Dr. Hakeem AI API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "explain": "/explain",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return health_check()


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():
    return get_model_info()


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
):
    """
    Standard classification endpoint.

    Multipart field:
        file

    Returns one best prediction or object_not_defined.
    """

    image_bytes = await read_image_file(
        file
    )

    try:

        logger.info(
            "Starting /predict request."
        )

        result = predict(
            image_bytes
        )

        logger.info(
            "Finished /predict successfully."
        )

        return result

    except ValueError as e:

        logger.warning(
            "Prediction validation error: %s",
            str(e),
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    except Exception as e:

        logger.error(
            "PREDICT FAILED:\n%s",
            traceback.format_exc(),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {str(e)}"
            ),
        ) from e

    finally:

        # Explicitly release the local reference.
        image_bytes = None


# ============================================================
# EXPLAIN / GRAD-CAM
# ============================================================

@app.post("/explain")
async def explain_endpoint(
    file: UploadFile = File(...),
    target_class: int | None = Query(
        default=None,
        description=(
            "Optional class index for Grad-CAM. "
            "If omitted, the predicted class is used."
        ),
    ),
    alpha: float = Query(
        default=0.45,
        ge=0.10,
        le=0.85,
        description=(
            "Heatmap overlay opacity."
        ),
    ),
):
    """
    Grad-CAM explainability endpoint.

    Multipart field:
        file

    Optional:
        target_class
        alpha

    Important:
        This endpoint performs prediction + Grad-CAM
        in ONE PyTorch pass.
    """

    image_bytes = await read_image_file(
        file
    )

    try:

        logger.info(
            "Starting /explain request | "
            "target_class=%s | alpha=%s",
            target_class,
            alpha,
        )

        result = explain(
            image_bytes=image_bytes,
            target_class=target_class,
            alpha=alpha,
        )

        logger.info(
            "Finished /explain request | "
            "status=%s",
            result.get("status"),
        )

        return result

    except ValueError as e:

        logger.warning(
            "Explain validation error: %s",
            str(e),
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    except FileNotFoundError as e:

        logger.error(
            "Required model file missing: %s",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except RuntimeError as e:

        logger.error(
            "GRAD-CAM RUNTIME ERROR:\n%s",
            traceback.format_exc(),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Grad-CAM runtime error: {str(e)}"
            ),
        ) from e

    except Exception as e:

        logger.error(
            "GRAD-CAM FAILED:\n%s",
            traceback.format_exc(),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Grad-CAM failed: {str(e)}"
            ),
        ) from e

    finally:

        image_bytes = None


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    import os
    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )
