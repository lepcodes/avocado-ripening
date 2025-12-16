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

MODEL_LOCAL_PATH = "models/avocado-model"
MLFLOW_TRACKING_URI = os.environ["MLFLOW_INTERNAL_URI"]


def load_model_into_memory():
    """
    Loads the model from the LOCAL disk into the global dictionary.
    """
    try:
        logger.info("🧠 Loading model into RAM from...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        tf.keras.config.enable_unsafe_deserialization()
        models["model"] = mlflow.tensorflow.load_model("models:/avocado-model@champion")
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
        load_model_into_memory()
        return {"message": "Model reloaded"}
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading model: {e}")


@app.post("/predict")
async def predict(
    image_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    image_name: Optional[str] = Form(None),
    storage_condition: StorageCondition = StorageCondition.Tam,
):
    """
    Make a prediction of ripeness of an avocado based on its image and storage condition.
    - image_file: Upload an image file directly. (Option 1)
    - image_url: Provide a URL to an image. (Option 2)
    - image_name: Provide the name of an image file in the 'images' directory. (Option 2)
    - storage_condition: Specify the storage condition (T10, T20, Tam). Default is 'Tam'.
    """

    # --- 1. Validation and Data Acquisition ---

    if image_file and (image_url or image_name):
        # Error if both methods are provided
        raise HTTPException(
            status_code=400,
            detail="Provide either 'image_file' OR ('image_url' and 'image_name'), not both.",
        )

    # --- 2. Image Loading and Processing ---

    try:
        if image_file:
            print(f"Processing uploaded file: {image_file.filename}")
            image = await validate_image_pil(image_file)

        elif image_url and image_name:
            print(f"Processing image from URL: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status()
            image_source = response.content
            image = Image.open(io.BytesIO(image_source))

        else:
            raise HTTPException(
                status_code=400,
                detail="Image data missing. Provide 'image_file' or both 'image_url' and 'image_name'.",
            )
    except HTTPException as e:
        # Re-raise HTTP exceptions to send proper client errors
        raise e
    except Exception as e:
        # Better logging for unexpected errors
        print(f"An unexpected error occurred during image processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    # --- 3. Image Transformation ---
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Process the storage condition
    storage_condition = np.array([mapping[storage_condition]])

    # --- 4. Model Prediction ---
    x = {"image_input": img_array, "condition_input": storage_condition}
    try:
        y = models["model"].predict(x)
        prediction = y[0][0]
        return PredictionResponse(
            prediction=[Prediction(estimated_days=float(prediction))]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
