# -*- coding: utf-8 -*-
"""
01_preprocess_data.py

This script performs two main preprocessing tasks:
1. Data Engineering: Loads the metadata, computes the time until overripening
   ('Shelf-life Days'), one-hot encodes the storage group, and saves the
   processed CSV metadata.
2. Image Preprocessing: Removes the background from all raw images using 'rembg'
   with GPU support (CUDA) for faster processing and saves the clean images.

It is designed to be run from the project root directory in a Cookiecutter
Data Science structure.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2

# Note: Ensure 'rembg' and 'opencv-python' are installed
# Note: For GPU support, 'onnxruntime-gpu' is required.
try:
    from rembg import new_session, remove

    HAS_REMBG = True
except ImportError:
    print("Warning: 'rembg' library not found. Image preprocessing will be skipped.")
    HAS_REMBG = False


# --- Configuration and Setup ---
# Define paths relative to the project root
# Metadata file paths
RAW_METADATA_PATH = (
    "data/external/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset.xlsx"
)
PROCESSED_METADATA_PATH = (
    "data/processed/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset.csv"
)

# Image folder paths
RAW_IMAGE_DIR = (
    "data/external/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset/"
)
PROCESSED_IMAGE_DIR = (
    "data/processed/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset/"
)

# Constants
SECONDS_IN_A_DAY = 24 * 60 * 60

# Initialize GPU session globally if rembg is available
if HAS_REMBG:
    provider = "CUDAExecutionProvider"
    try:
        print(f"Attempting to create a rembg session with {provider}...")
        # Initializing the session here for single load across all images
        GPU_SESSION = new_session("u2net", providers=[provider])
        print("rembg session initialized successfully.")
    except Exception as e:
        print(f"Error initializing rembg session with CUDA: {e}")
        print("Falling back to default CPU provider.")
        GPU_SESSION = new_session("u2net")


# ==============================================================================
#                      PART 1: METADATA PREPROCESSING
# ==============================================================================


# --- Data Loading ---
def load_metadata(file_path):
    """Loads the raw metadata from an Excel file."""
    print(f"\n--- METADATA PREPROCESSING ---\nLoading metadata from: {file_path}")
    try:
        data = pd.read_excel(file_path)
        # Ensure 'Time Stamp' is in datetime format immediately
        data["Time Stamp"] = pd.to_datetime(data["Time Stamp"])
        print(f"Metadata loaded successfully. Initial shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {file_path}. Please check the path.")
        return None


# --- Feature Engineering: Time Until Overripe ---
def compute_overripeness_time(time_stamp, overripe_time_stamp):
    """Calculates the timedelta between the current timestamp and the first overripening timestamp."""
    if pd.isna(overripe_time_stamp):
        return np.nan

    # Timestamps are already in datetime format from load_metadata
    dif = overripe_time_stamp - time_stamp

    # If the current time is after the overripe time, the remaining time is 0
    return pd.Timedelta(0) if dif < pd.Timedelta(0) else dif


def calculate_shelf_life(data):
    """Calculates the 'Overripening Day', 'Overripening Time Stamp', and the final 'Shelf-life Days' feature."""
    print(
        "Calculating first day/timestamp of overripening (Ripening Index Classification == 5)..."
    )

    # Filter for all entries marked as overripe (Index 5)
    overripe_days = data[data["Ripening Index Classification"] == 5]

    # Find the minimum day and corresponding earliest time stamp for overripening for each sample
    first_overripe_data = overripe_days.groupby("Sample")[
        ["Day of Experiment", "Time Stamp"]
    ].min()

    # Map the data back to the main DataFrame
    data["Overripening Day"] = data["Sample"].map(first_overripe_data["Day of Experiment"])
    data["Overripening Time Stamp"] = data["Sample"].map(first_overripe_data["Time Stamp"])

    print("Computing 'Time Unitl Overripening'...")
    # Apply the function to compute the time difference
    data["Time Unitl Overripening"] = data.apply(
        lambda x: compute_overripeness_time(x["Time Stamp"], x["Overripening Time Stamp"]),
        axis=1,
        # Use 'raw=False' to pass Series objects to the lambda, which is needed here
    )

    print("Converting timedelta to 'Shelf-life Days' (float)...")
    # Convert the timedelta object into total days (float)
    data["Shelf-life Days"] = data["Time Unitl Overripening"].dt.total_seconds() / SECONDS_IN_A_DAY

    return data


# --- Feature Encoding and Cleanup ---
def process_metadata_features(data):
    """Performs one-hot encoding on 'Storage Group', drops the original column, and handles NaNs."""

    print("One-hot encoding 'Storage Group' feature...")
    # One-hot encode the categorical 'Storage Group' column
    data = pd.concat([data, pd.get_dummies(data["Storage Group"], dtype=int)], axis=1)

    print("Dropping original 'Storage Group' column...")
    # Drop the original 'Storage Group' column as it's been encoded
    data.drop(["Storage Group"], axis=1, inplace=True)

    initial_rows = data.shape[0]
    print("Dropping NaN values (samples that did not overripen in the experiment)...")
    data.dropna(inplace=True)

    rows_dropped = initial_rows - data.shape[0]
    print(f"Dropped {rows_dropped} rows. Final metadata shape: {data.shape}")

    return data


# ==============================================================================
#                      PART 2: IMAGE PREPROCESSING
# ==============================================================================


def remove_background_from_image(image_path, session):
    """Reads image, removes background using rembg, and fills the background with white."""

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    # rembg works with RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Remove background to get an RGBA image
    output_rgba = remove(img_rgb, session=session)

    # Separate RGB and Alpha channels
    output_rgb_only = output_rgba[:, :, :3]
    alpha = output_rgba[:, :, 3] / 255.0

    # Create a white background layer
    # Apply anti-aliased alpha blending to fill the background with white
    white_background = (1.0 - alpha[:, :, None]) * 255
    final_image_rgb = (alpha[:, :, None] * output_rgb_only + white_background).astype("uint8")

    # Convert back to BGR for OpenCV saving
    final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)

    return final_image_bgr


def process_images(input_dir, output_dir, session):
    """Walks through the input directory, processes images, and saves them to the output directory."""

    if not HAS_REMBG:
        print("\nImage processing skipped due to missing 'rembg' library.")
        return

    print(f"\n--- IMAGE PREPROCESSING ---\nStarting image processing from '{input_dir}'")
    print(f"Processed images will be saved to '{output_dir}'")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all files to process for the tqdm progress bar
    all_files = os.listdir(input_dir)
    image_files = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for filename in tqdm(image_files, desc="Removing background from images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Process and save the image
        processed_image = remove_background_from_image(input_path, session)
        if processed_image is not None:
            cv2.imwrite(output_path, processed_image)

    print("Image preprocessing completed! ✨")


# ==============================================================================
#                             MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":

    # ----------------------------------------------
    # 1. METADATA PROCESSING
    # ----------------------------------------------

    # Load and process the metadata
    metadata = load_metadata(RAW_METADATA_PATH)

    if metadata is not None:
        # Compute the target variable ('Shelf-life Days')
        metadata = calculate_shelf_life(metadata)

        # Process categorical features and handle missing data
        metadata = process_metadata_features(metadata)

        # Save the processed metadata
        print(f"Saving processed metadata to: {PROCESSED_METADATA_PATH}")
        metadata.to_csv(PROCESSED_METADATA_PATH, index=False)
        print("Metadata processing complete.")

    # ----------------------------------------------
    # 2. IMAGE PROCESSING
    # ----------------------------------------------

    # Process the images (if rembg is available)
    if HAS_REMBG:
        process_images(RAW_IMAGE_DIR, PROCESSED_IMAGE_DIR, GPU_SESSION)
    else:
        print("\nSkipping image processing. Install 'rembg' to enable this feature.")
