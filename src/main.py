from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from tensorflow import keras
from enum import Enum
from typing import List
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import os


class StorageCondition(str, Enum):
    T10 = "T10"
    T20 = "T20"
    Tam = "Tam"


class Prediction(BaseModel):
    estimated_days: float


class PredictionResponse(BaseModel):
    prediction: List[Prediction]


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE on application startup
    print("Application startup: Loading model...")

    # The path inside the container where models are mounted
    model_dir = "/app/models"

    model_name = os.getenv("MODEL_NAME")

    if not model_name:
        raise ValueError("Environment variable 'MODEL_NAME' is not set.")

    model_path = os.path.join(model_dir, model_name)
    try:
        models["model"] = keras.models.load_model(model_path)
        print(f"Successfully loaded model from: {model_path}")
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at: {model_path}")

    yield

    # This code runs ONCE on application shutdown
    print("Application shutdown: Cleaning up resources...")
    models.clear()


app = FastAPI(lifespan=lifespan)

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


@app.post("/predict")
async def predict(
    image_file: UploadFile = File(...), storage_condition: StorageCondition = StorageCondition.Tam
):
    """A dummy prediction endpoint that returns image size."""

    # Process the uploaded image file
    image = await validate_image_pil(image_file)
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Process the storage condition
    storage_condition = np.array([mapping[storage_condition]])

    # Make the prediction
    x = {"image_input": img_array, "condition_input": storage_condition}
    try:
        y = models["model"].predict(x)
        prediction = y[0][0]
        return PredictionResponse(prediction=[Prediction(estimated_days=float(prediction))])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
