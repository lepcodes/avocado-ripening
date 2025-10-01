from fastapi import FastAPI
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from contextlib import asynccontextmanager
from tensorflow import keras
from enum import Enum
from typing import List
from pydantic import BaseModel
from PIL import Image
import numpy as np
import requests
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
    model_dir = os.getenv("MODEL_DIR")
    if not model_dir:
        raise ValueError("Environment variable 'MODEL_DIR' is not set.")

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
        return PredictionResponse(prediction=[Prediction(estimated_days=float(prediction))])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
