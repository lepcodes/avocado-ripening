import json
import os
import shutil
import zipfile

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config
from dotenv import load_dotenv
from prefect import flow, task
from prefect.logging import get_run_logger

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()


def compute_overripeness_time(time_stamp, overripe_time_stamp):
    if pd.isna(overripe_time_stamp):
        return np.nan
    dif = pd.Timestamp(overripe_time_stamp) - pd.Timestamp(time_stamp)
    if dif < pd.Timedelta(0):
        return pd.Timedelta(0)
    else:
        return dif


@task
def fetch_data(
    bucket_name: str = "ampere-instance-bucket",
    file_key: str = "dataset.zip",
    data_path: str = "data",  # Change this to docker local path
):
    """
    Downloads the dataset from the bucket and extracts it to the local path.
    """
    logger = get_run_logger()
    raw_dir = os.path.join(data_path, "raw")
    zip_path = os.path.join(raw_dir, file_key)

    # Caching (Check if the file already exists in the local path)
    logger.info("Caching data... (Check raw data already exists)")
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists(os.path.join(raw_dir, "data.csv")):
        logger.info("Raw data already exists. Skipping data acquisition.")
        return raw_dir

    # Fetch the data from the OCI bucket
    logger.info(f"Fetching data from OCI bucket: {bucket_name}")
    local_path = os.path.join(data_path, "raw")
    zip_path = os.path.join(local_path, file_key)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("OCI_ENDPOINT_URL"),
        config=Config(signature_version="s3v4"),
    )
    s3.download_file(bucket_name, file_key, zip_path)

    # Extract the data from the ZIP file
    logger.info("Extracting data from ZIP file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(local_path)

    os.remove(zip_path)

    # Data acquisition complete
    logger.info("Data acquisition complete.")
    return raw_dir


@task
def preprocess_data(raw_data_path: str):
    """
    Preprocess Data
    """
    logger = get_run_logger()

    # Go up one directory
    dataset_path = os.path.dirname(raw_data_path)
    processed_dir = os.path.join(dataset_path, "processed")

    # Caching (Check if the file already exists in the local path)
    logger.info("Caching data... (Check processed data already exists)")
    if os.path.exists(os.path.join(processed_dir, "data.csv")):
        logger.info("Processed data already exists. Skipping preprocessing.")
        return processed_dir

    # Read the data from the CSV file
    logger.info("Reading raw data...")
    data = pd.read_csv(os.path.join(raw_data_path, "data.csv"))

    # Preprocess the data
    logger.info("Preprocessing data...")
    overripe_days = data[data["Ripening Index Classification"] == 5]
    first_overripe_day = overripe_days.groupby("Sample")[["Day of Experiment", "Time Stamp"]].min()
    data["Overripening Day"] = data["Sample"].map(first_overripe_day["Day of Experiment"])
    data["Overripening Time Stamp"] = data["Sample"].map(first_overripe_day["Time Stamp"])
    data["Time Unitl Overripening"] = data[["Time Stamp", "Overripening Time Stamp"]].apply(
        lambda x: compute_overripeness_time(x["Time Stamp"], x["Overripening Time Stamp"]), axis=1
    )
    seconds_in_a_day = 24 * 60 * 60
    data["Shelf-life Days"] = data["Time Unitl Overripening"].dt.total_seconds() / seconds_in_a_day
    data = pd.concat([data, pd.get_dummies(data["Storage Group"], dtype=int)], axis=1)
    data.drop(
        [
            "Time Stamp",
            "Sample",
            "Day of Experiment",
            "Ripening Index Classification",
            "Storage Group",
        ],
        axis=1,
        inplace=True,
    )
    data.drop(
        ["Overripening Day", "Overripening Time Stamp", "Time Unitl Overripening"],
        axis=1,
        inplace=True,
    )
    data.dropna(inplace=True)
    data["Absolute Path"] = data["File Name"].apply(lambda x: os.path.join(raw_data_path, "images", x + ".jpg"))

    mask_existing_images = data["Absolute Path"].apply(os.path.exists)
    logger.info(f"Unexisting images: {data['Absolute Path'].count() - mask_existing_images.sum()}")
    data = data[mask_existing_images].copy()
    data.drop(["Absolute Path"], axis=1, inplace=True)
    # Final column reordering
    logger.info("Final column reordering...")
    target_columns = [
        "File Name",       # 1st
        "T10",             # 2nd
        "T20",             # 3rd
        "Tam",             # 4th
        "Shelf-life Days"  # 5th
    ]
    data = data.reindex(columns=target_columns, fill_value=0)
    os.makedirs(processed_dir, exist_ok=True)
    data.to_csv(os.path.join(processed_dir, "data.csv"), index=False)

    # Preprocessing complete
    logger.info("Preprocessing complete.")
    return processed_dir

@task
def upload_preprocessed_data(processed_dir, raw_dir):
    logger = get_run_logger()

    logger.info("Creating staging area for dataset upload...")
    staging_dir = os.path.join(os.path.dirname(processed_dir), "staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.join(staging_dir, "images"), exist_ok=True)

    logger.info("Copying dataset images to staging area...")
    data = pd.read_csv(os.path.join(processed_dir, "data.csv"))
    origin_images_dir = os.path.join(raw_dir, "images")
    stagin_images_dir = os.path.join(staging_dir, "images")
    count = 0
    for file_name in data["File Name"]:
        src_file = os.path.join(origin_images_dir, file_name + ".jpg")
        dst_file = os.path.join(stagin_images_dir, file_name + ".jpg")
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            count += 1
        else:
            logger.warning(f"❌ File listed in dataset but not found: {src_file}")
    logger.info(f"✅ Copied {count} images to staging area.")

    logger.info("Copying dataset csv to staging area...")
    origin_csv_path = os.path.join(processed_dir, "data.csv")
    staging_csv_path = os.path.join(staging_dir, "data.csv")
    shutil.copy(origin_csv_path, staging_csv_path)

    logger.info("Creating dataset metadata file...")
    user = "luispreciado99"
    dataset_slug = "avocado-ripening-dataset"
    dataset_id = f"{user}/{dataset_slug}"
    metadata = {
        "title": "Avocado Ripening Dataset",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(staging_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Authenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error(f"⚠️ Error autenticando con Kaggle: {e}")
        raise e

    logger.info("Uploading dataset to Kaggle Datasets...")
    message = "Uploading dataset structure (CSV + Images)"
    try:
        logger.info("Attempting to update dataset version...")
        api.dataset_create_version(
            staging_dir, 
            version_notes=message, 
            dir_mode='zip',
            quiet=False
        )
        logger.info("✅ New dataset version created successfully.")
        
    except Exception as e:
        error_str = str(e).lower()
        if any(x in error_str for x in ["404", "not found", "403", "forbidden"]):
            logger.info(f"⚠️ Dataset not updated (Error {error_str}).")
            logger.info("🆕 Trying to create it from scratch...")
            try:
                api.dataset_create_new(
                    staging_dir, 
                    dir_mode='zip', 
                    public=False,
                    quiet=False
                )
                logger.info("✅ Dataset created successfully.")
            except Exception as e2:
                logger.error(f"❌ Fatal error trying to create: {e2}")
                raise e2
                
        else:
            logger.error(f"❌ Unexpected error not handled: {e}")
            raise e


@task
def train_model(data, dataset_path):
    data.to_csv(dataset_path + "/data.csv", index=False)
    return data


@task
def evaluate_model(data):
    print("Evaluating model")
    return data


@task
def deploy_model(data):
    print("Deploying model")
    return data


@flow
def test_flow():
    logger = get_run_logger()

    raw_data_path = fetch_data()
    logger.info(f"Dataset path: {raw_data_path}")

    preprocessed_data_path = preprocess_data(raw_data_path)
    logger.info(f"Preprocessed data path: {preprocessed_data_path}")

    upload_preprocessed_data(preprocessed_data_path, raw_data_path)
    return raw_data_path


if __name__ == "__main__":
    test_flow()
