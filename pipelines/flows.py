from prefect import flow, task
import pandas as pd
import numpy as np
import shutil
import os


def compute_overripeness_time(time_stamp, overripe_time_stamp):
    if pd.isna(overripe_time_stamp):
        return np.nan
    dif = pd.Timestamp(overripe_time_stamp) - pd.Timestamp(time_stamp)
    if dif < pd.Timedelta(0):
        return pd.Timedelta(0)
    else:
        return dif


# ------------------------------------------------------------------------------
# Tasks 1 - Get data
# ------------------------------------------------------------------------------


@task
def get_data():
    dummy_origin = (
        "C:/Users/Luis/Documents/ML-AI-Projects/avocado-ripening/data/external/dataset_batch_1"
    )
    dummy_destination = (
        "C:/Users/Luis/Documents/ML-AI-Projects/avocado-ripening/data/raw/pipeline_test"
    )

    # Copy the data from the source to the destination
    shutil.copytree(dummy_origin + "/images", dummy_destination + "/images", dirs_exist_ok=True)
    shutil.copy(dummy_origin + "/data.csv", dummy_destination + "/data.csv")
    data = pd.read_csv(dummy_destination + "/data.csv")
    return data, dummy_destination


# ------------------------------------------------------------------------------
# Tasks 2 - Preprocess data
# ------------------------------------------------------------------------------


@task
def preprocess_data(data, dataset_path):
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

    data["File Path"] = data["File Name"].apply(lambda x: dataset_path + "/images/" + x + ".jpg")
    data.drop(["File Name"], axis=1, inplace=True)

    mask_existing_images = data["File Path"].apply(os.path.exists)
    data = data[mask_existing_images].copy()
    return data, dataset_path


# ------------------------------------------------------------------------------
# Tasks 3 - Train model
# ------------------------------------------------------------------------------


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
    # data, dummy_destination = get_data()
    # data, dataset_path = preprocess_data(data, dummy_destination)
    # train_model(data, dataset_path)
    return "Model trained"


if __name__ == "__main__":
    test_flow()
