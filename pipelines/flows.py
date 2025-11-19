from prefect import flow, task
from prefect.logging import get_run_logger
import pandas as pd
import numpy as np
import boto3
from botocore.client import Config
import zipfile
from dotenv import load_dotenv
import os

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

    data["File Path"] = data["File Name"].apply(lambda x: raw_data_path + "/images/" + x + ".jpg")
    data.drop(["File Name"], axis=1, inplace=True)

    mask_existing_images = data["File Path"].apply(os.path.exists)
    data = data[mask_existing_images].copy()

    # Final column reordering
    logger.info("Final column reordering...")
    target_columns = [
        "File Path",       # 1st
        "T10",             # 2nd
        "T20",             # 3rd
        "Tam",             # 4th
        "Shelf-life Days"  # 5th
    ]
    data = data.reindex(columns=target_columns, fill_value=0)
    data.to_csv(os.path.join(processed_dir, "data.csv"), index=False)

    # Preprocessing complete
    logger.info("Preprocessing complete.")
    return processed_dir


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
    data_path = fetch_data()
    logger.info(f"Dataset path: {data_path}")
    preprocessed_data_path = preprocess_data(data_path)
    logger.info(f"Preprocessed data path: {preprocessed_data_path}")
    return data_path


if __name__ == "__main__":
    test_flow()
