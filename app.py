"""FastAPI REST API for Banana Ripeness Detection inference."""

import io
import os
from typing import Dict

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet10_distilled_student_model")
CLASS_NAMES = ["Overripe", "Ripe", "Unripe", "Rotten"]
IMG_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Banana Ripeness Detection API",
    description=(
        "REST API for detecting banana ripeness using a knowledge-distilled "
        "ResNet10 student model trained on the Banana Ripeness Classification "
        "dataset."
    ),
    version="1.0.0",
)

# Load model at startup
_model = None


@app.on_event("startup")
def load_model():
    """Load the Keras model on application startup."""
    import tensorflow as tf

    global _model
    if os.path.exists(MODEL_PATH):
        _model = tf.keras.models.load_model(MODEL_PATH)
    else:
        # Allow the app to start without a model for testing
        _model = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Health check endpoint.

    Returns:
        dict: Status and whether the model is loaded.
    """
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict the ripeness of a banana from an uploaded image.

    Args:
        file: An image file (JPEG, PNG, etc.) uploaded via multipart form.

    Returns:
        PredictionResponse: Predicted class, confidence, and per-class
            probabilities.

    Raises:
        HTTPException 503: If the model is not loaded.
        HTTPException 400: If the uploaded file cannot be read as an image.
    """
    import tensorflow as tf

    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    logits = _model.predict(img_array, verbose=0)
    probabilities = tf.nn.softmax(logits[0]).numpy()
    predicted_idx = int(np.argmax(probabilities))

    return PredictionResponse(
        predicted_class=CLASS_NAMES[predicted_idx],
        confidence=float(probabilities[predicted_idx]),
        probabilities={
            name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)
        },
    )


@app.get("/classes")
def get_classes():
    """Return the list of supported ripeness classes.

    Returns:
        dict: Dictionary with a ``classes`` key.
    """
    return {"classes": CLASS_NAMES}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
