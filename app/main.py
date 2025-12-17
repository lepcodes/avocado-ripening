import io
import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import mlflow

load_dotenv()


class StorageCondition(str, Enum):
    T10 = "T10"
    T20 = "T20"
    Tam = "Tam"


class Prediction(BaseModel):
    estimated_days: float


class PredictionResponse(BaseModel):
    prediction: List[Prediction]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

models = {}

MODEL_LOCAL_PATH = "models/avocado-model.keras"
MLFLOW_TRACKING_URI = os.environ["MLFLOW_INTERNAL_URI"]
DEV_MODE = os.environ.get("DEV_MODE", "False").lower() == "true"


def load_model_into_memory(force_download=False):
    """
    Loads the model from Mlflow into the glob.
    """
    try:
        tf.keras.config.enable_unsafe_deserialization()
        if (not force_download and os.path.exists(MODEL_LOCAL_PATH)) and DEV_MODE:
            logger.info(f"🧠 Loading model from local path: {MODEL_LOCAL_PATH}")
            models["model"] = tf.keras.models.load_model(MODEL_LOCAL_PATH)
            logger.info("✅ Model loaded from local disk.")
            return

        logger.info("🧠 Loading model into RAM from...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.tensorflow.load_model("models:/avocado-model@champion")
        models["model"] = model
        os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
        model.save(MODEL_LOCAL_PATH)
        logger.info("✅ Model loaded into memory.")
    except Exception as e:
        logger.error(f"❌ Error loading model into memory: {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    logger.info(f"MLflow Internal URI:d {MLFLOW_TRACKING_URI}")
    load_model_into_memory()
    yield
    logger.info("Application shutdown: Cleaning up resources...")
    models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


mapping = {
    StorageCondition.T10: [1, 0, 0],
    StorageCondition.T20: [0, 1, 0],
    StorageCondition.Tam: [0, 0, 1],
}


async def validate_image_pil(file: UploadFile):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return Image.open(io.BytesIO(contents))


@app.get("/")
async def root():
    """A simple root endpoint."""
    return {"message": "Welcome to avocado-ripening API"}


@app.get("/health")
async def health():
    """A simple health check endpoint."""
    try:
        logger.info("Checking model health...")
        dummy_img = np.zeros((1, 224, 224, 3))
        dummy_cond = np.zeros((1, 3))
        _ = models["model"].predict(
            {"image_input": dummy_img, "condition_input": dummy_cond}
        )
        return {"message": "Model is healthy"}
    except Exception as e:
        logger.error(f"Error checking model health: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking model health: {e}")


@app.post("/reload-model")
async def reload_model():
    """Reload the model from MLFlow."""
    try:
        logger.info("Reloading model from MLFlow...")
        load_model_into_memory(force_download=True)
        return {"message": "Model reloaded"}
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading model: {e}")


@app.post("/predict")
async def predict(
    image_files: Optional[List[UploadFile]] = File(None),
    image_urls: Optional[List[str]] = Form(None),
    storage_conditions: List[StorageCondition] = Form(...),
):
    """
    Make a prediction of ripeness of an avocado based on its image and storage condition.
    - image_file: Upload an image file directly. (Option 1)
    - image_url: Provide a URL to an image. (Option 2)
    - storage_condition: Specify the storage condition (T10, T20, Tam). Default is 'Tam'.
    """

    # --- 1. Image Loading ---
    images = []
    if image_files:
        for image in image_files:
            image = await validate_image_pil(image)
            images.append(image)

    if image_urls:
        for url in image_urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                image_source = response.content
                image = Image.open(io.BytesIO(image_source))
                images.append(image)
            except Exception as e:
                logger.error(f"❌ Error loading image from URL: {e}")
                raise HTTPException(status_code=400, detail=f"Error loading image from URL {url}: {e}")

    if not images:
        raise HTTPException(status_code=400, detail="No images provided.")

    if not storage_conditions:
        raise HTTPException(status_code=400, detail="No storage conditions provided.")

    if len(storage_conditions) != len(images):
        raise HTTPException(
            status_code=400,
            detail="Number of storage conditions and images do not match.",
        )

    # --- 3. Image Transformation ---
    processed_images = []
    for image in images:
        try:
            image = image.resize((224, 224))
            img_array = np.array(image)
            processed_images.append(img_array)
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    # --- 4. Storage Conditions ---
    processed_conditions = []
    for storage_condition in storage_conditions:
        storage_condition = np.array(mapping[storage_condition])
        processed_conditions.append(storage_condition)

    # --- 5. Batching ---
    batch_images = np.array(processed_images)
    batch_conditions = np.array(processed_conditions)
    logger.info(f"✅ Batched images: {batch_images.shape}")
    logger.info(f"✅ Batched conditions: {batch_conditions.shape}")

    # --- 6. Model Prediction ---
    x = {"image_input": batch_images, "condition_input": batch_conditions}
    try:
        predictions = models["model"].predict(x)

        logger.info(f"✅ Prediction: {predictions[:, 0]}")
        output = []
        for prediction in predictions[:, 0]:
            output.append(Prediction(estimated_days=float(prediction)))
        return PredictionResponse(prediction=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
