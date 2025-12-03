import json
import os
import shutil
import time
import zipfile
import boto3
import numpy as np
import pandas as pd
import nbformat as nbf
from botocore.client import Config
from dotenv import load_dotenv
from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.states import Failed
from prefect.context import get_run_context
from mlflow.tracking import MlflowClient

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi


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
    logger.info("📋 Beginning data acquisition...")

    # Caching (Check if the file already exists in the local path)
    logger.info("\tCaching data... (Check raw data already exists)")
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists(os.path.join(raw_dir, "data.csv")):
        logger.info("Raw data already exists. Skipping data acquisition.")
        return raw_dir

    # Fetch the data from the OCI bucket
    logger.info(f"\tFetching data from OCI bucket: {bucket_name}")
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
    logger.info("\tExtracting data from ZIP file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(local_path)

    os.remove(zip_path)

    # Data acquisition complete
    logger.info("\tData acquisition complete.")
    return raw_dir


@task
def preprocess_data(raw_data_path: str):
    """
    Preprocess Data
    """
    logger = get_run_logger()
    logger.info("📋 Beginning data preprocessing...")

    # Go up one directory
    dataset_path = os.path.dirname(raw_data_path)
    processed_dir = os.path.join(dataset_path, "processed")

    # Caching (Check if the file already exists in the local path)
    logger.info("\tCaching data... (Check processed data already exists)")
    if os.path.exists(os.path.join(processed_dir, "data.csv")):
        logger.info("Processed data already exists. Skipping preprocessing.")
        return processed_dir

    # Read the data from the CSV file
    logger.info("\tReading raw data...")
    data = pd.read_csv(os.path.join(raw_data_path, "data.csv"))

    # Preprocess the data
    logger.info("\tPreprocessing data...")
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
    logger.info(f"\tUnexisting images: {data['Absolute Path'].count() - mask_existing_images.sum()}")
    data = data[mask_existing_images].copy()
    data.drop(["Absolute Path"], axis=1, inplace=True)
    # Final column reordering
    logger.info("\tFinal column reordering...")
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
    logger.info("\tPreprocessing complete.")
    return processed_dir

@task
def upload_preprocessed_data(processed_dir, raw_dir):
    logger = get_run_logger()
    logger.info("📋 Beginning dataset upload...")

    logger.info("\tCreating staging area for dataset upload...")
    staging_dir = os.path.join(os.path.dirname(processed_dir), "staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.join(staging_dir, "images"), exist_ok=True)

    logger.info("\tCopying dataset images to staging area...")
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
    logger.info(f"\tCopied {count} images to staging area.")

    logger.info("\tCopying dataset csv to staging area...")
    origin_csv_path = os.path.join(processed_dir, "data.csv")
    staging_csv_path = os.path.join(staging_dir, "data.csv")
    shutil.copy(origin_csv_path, staging_csv_path)

    logger.info("\tCreating dataset metadata file...")
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

    logger.info("\tAuthenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error(f"⚠️ Error authenticating with Kaggle: {e}")
        raise e

    logger.info("\tUploading dataset to Kaggle Datasets...")
    message = "Uploading dataset structure (CSV + Images)"
    try:
        logger.info("Attempting to update dataset version...")
        api.dataset_create_version(
            staging_dir, 
            version_notes=message, 
            dir_mode='zip',
            quiet=False
        )
        logger.info("\t✅ New dataset version created successfully.")
        
    except Exception as e:
        error_str = str(e).lower()
        if any(x in error_str for x in ["404", "not found", "403", "forbidden"]):
            logger.info(f"⚠️ Dataset not updated (Error {error_str}).")
            logger.info("\t🆕 Trying to create it from scratch...")
            try:
                api.dataset_create_new(
                    staging_dir, 
                    dir_mode='zip', 
                    public=False,
                    quiet=False
                )
                logger.info("\t✅ Dataset created successfully.")
            except Exception as e2:
                logger.error(f"❌ Fatal error trying to create: {e2}")
                raise e2
                
        else:
            logger.error(f"❌ Unexpected error not handled: {e}")
            raise e


@task
def train_model( 
    run_id: str,
    model_name: str,
    experiment_name: str,
    run_name: str,
):
    logger = get_run_logger()
    logger.info("📋 Beginning model training...")
    KAGGLE_USERNAME = "luispreciado99"
    PROJECT_SLUG = "avocado-ripening-notebook"
    DATASET_SLUG = "avocado-ripening-dataset"
    NOTEBOOK_PATH = "notebooks/training-notebook-template.ipynb"

    logger.info("\tAuthenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    logger.info("\tCreating metadata for kernel...")
    notebook_abs_path = os.path.abspath(NOTEBOOK_PATH)
    notebook_parent_dir = os.path.dirname(notebook_abs_path)
    notebook_staged_path = os.path.join(notebook_parent_dir, "training-notebook-staged.ipynb")
    kernel_id = f"{KAGGLE_USERNAME}/{PROJECT_SLUG}"
    dataset_id = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"
    if not os.path.exists(notebook_abs_path):
        raise FileNotFoundError(f"Could not find notebook at: {notebook_abs_path}")
    metadata = {
        "id": kernel_id,
        "title": PROJECT_SLUG,
        "code_file": os.path.basename(notebook_staged_path),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [], # [dataset_id],
        "kernel_sources": [],
        "competition_sources": []
    }
    with open(os.path.join(notebook_parent_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("\tInjecting code cells...")
    notebook = nbf.read(notebook_abs_path, as_version=4)
    if notebook.nbformat_minor < 5:
        notebook.nbformat_minor = 5
    parameters = f"""
# ------ HYPERPARAMETERS ------
BATCH_SIZE = 12
EPOCHS = 10
LEARNING_RATE = 0.001

# ------ PREFECT_RUN_ID ------
PREFECT_RUN_ID = "{run_id}"
EXPERIMENT_NAME = "{experiment_name}"
RUN_NAME = "{run_name}"
"""
    new = nbf.v4.new_code_cell(parameters)
    notebook.cells.insert(2, new)
    if notebook.nbformat_minor < 5:
        notebook.nbformat_minor = 5
    nbf.write(notebook, notebook_staged_path)

    logger.info("\tPushing kernel...")
    try:
        api.kernels_push(notebook_parent_dir)
        logger.info("✅ Kernel pushed successfully.")
        os.remove(notebook_staged_path)
    except Exception as e:
        logger.error(f"❌ Error pushing kernel: {e}")
        raise e

    logger.info("\tPolling status loop...")
    while True:
        try:
            response = api.kernels_status(kernel_id)
            raw_status = str(response.status)
            status_clean = raw_status.split('.')[-1].lower()
            logger.info(f"\tStatus: {status_clean.upper()} (Raw: {raw_status}) | Time: {time.strftime('%H:%M:%S')}")

            if status_clean == 'complete':
                logger.info("\t✅ Kernel finished successfully!")
                break
            
            elif status_clean == 'error':
                failure_msg = str(response.failureMessage)
                raise Exception(f"\t❌ Kaggle Training Failed: {failure_msg} | Time: {time.strftime('%H:%M:%S')}")

            elif status_clean in ['queued', 'running', 'starting', 'canceling']:
                time.sleep(10)
            
            else:
                logger.warning(f"\t⚠️ Unknown state {status_clean}. Keep waiting...")
                time.sleep(20)

        except Exception as e:
            logger.error(f"\t❌ Error checking status: {e}")
            raise Failed(message=f"Error checking status: {e}")
    return run_id


@task
def evaluate_model(
    run_id: str,
    model_name: str,
    experiment_name: str,
):
    logger = get_run_logger()
    logger.info("📋 Beginning model evaluation...")
    client = MlflowClient("https://mlflow.lepcodes.com")
    
    # Search Experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
    
    # Get Challenger Run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.prefect_run_id = '{run_id}'"
    )
    challenger_run = runs[0]
    challenger_metric = challenger_run.data.metrics.get('accuracy_test', 0)
    challenger_run_id = challenger_run.info.run_id
    logger.info(f"\tChallenger found: {challenger_metric}")

    # Get Current Champion 
    try:
        logger.info("\tGetting Champion...")
        champion_info = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(champion_info.run_id)
        champion_metric = champion_run.data.metrics.get('accuracy_test', 0)
        logger.info(f"\tChampion found: {champion_metric}")
    except Exception as e:
        logger.info(f"\tNo Champion found: {e}")
        champion_metric = 0

    # Update Champion
    if challenger_metric > champion_metric:
        # Register Champion to Model Registry
        logger.info("\tPromotion! Updating Champion...")
        try:
            client.create_registered_model(model_name)
        except Exception as e:
            logger.info(f"\tChampion already registered: {e}")

        try:
            challenger_version = client.create_model_version(
                name=model_name,
                source=f"runs:/{challenger_run_id}/{model_name}",
                run_id=challenger_run_id
            )
            client.set_registered_model_alias(model_name, "champion", challenger_version.version)
            logger.info(f"\tChampion registered: {challenger_version.version}")
        except Exception as e:
            logger.info(f"\tError registering Champion: {e}")
    else:
        logger.info("\tNo Promotion. Keeping Champion...")
    return

@task
def deploy_model(data):
    print("Deploying model")
    return data


@flow
def test_flow(
    run_name: str,
    model_name: str = "avocado-model",
    experiment_name: str = "Avocado Ripening Experiment",
):
    logger = get_run_logger()

    # # raw_data_path = fetch_data()

    # # preprocessed_data_path = preprocess_data(raw_data_path)

    # # upload_preprocessed_data(preprocessed_data_path, raw_data_path)

    # Get Run ID
    prefect_run_id = get_run_context().flow_run.id
    logger.info(f"Run ID: {prefect_run_id}")

    train_model(
        run_id=prefect_run_id,
        model_name=model_name,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    evaluate_model(
        run_id=prefect_run_id, 
        model_name=model_name,
        experiment_name=experiment_name,
    )
    return


if __name__ == "__main__":
    test_flow.serve(name="my-first-deployment")
