# 🥑 Avocado Ripening Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project is a Deep Learning solution designed to predict the ripening stage of avocados based on image data. It leverages modern MLOps practices, including **Prefect** for orchestration, **MLFlow** for experiment tracking and model registry, and **Docker** for containerization.

## Tech Stack

*   **Python**: Primary programming language.
*   **TensorFlow/Keras**: Deep Learning framework for model development.
*   **Prefect**: Workflow orchestration for data processing, training, and deployment pipelines.
*   **MLFlow**: Platform for managing the MLOps lifecycle (experiment tracking, model registry).
*   **FastAPI**: For building the prediction API.
*   **Docker**: Containerization for reproducible environments.
*   **AWS S3 & RDS**: Cloud storage for artifacts and metadata.

## Project Organization

```
.
├── .dockerignore
├── .gitattributes
├── .gitignore
├── .vscode/
├── app/
│   ├── __init__.py
│   ├── main.py                <- FastAPI application entry point
│   └── ...
├── docker-compose.override.yaml <- Development overrides (Local)
├── docker-compose.yaml          <- Base/Production Docker composition
├── Dockerfile
├── Makefile
├── prefect.yaml                 <- Prefect deployment configuration
├── data/                        <- Data directory
├── mlartifacts/                 <- Local artifact storage
├── mlflow/                      <- MLFlow server configuration
├── models/                      <- Local model storage
├── notebooks/                   <- Jupyter notebooks for exploration & training
├── pipelines/
│   ├── flows.py                 <- Prefect flows and tasks definition
│   └── ...
└── src/                         <- Shared source code
```

## Core Components

### 🚀 Inference API (`avocado-api`)

The API is built with **FastAPI** and serves the trained model.

*   **Model Loading**: On startup, it connects to the MLFlow Model Registry and downloads the model tagged with the `@champion` alias.
*   **Endpoints**:
    *   `POST /predict`: Accepts an image (file or URL) and storage condition to predict ripeness days.
    *   `POST /reload-model`: Triggers a hot-reload of the model from MLFlow without restarting the container. This is called automatically by the Prefect pipeline after a successful model promotion.
    *   `GET /health`: Checks if the model is loaded and responsive.

### 🧪 MLFlow Server (`mlflow`)

Acts as the central hub for the machine learning lifecycle.

*   **Experiment Tracking**: Logs parameters, metrics, and artifacts from training runs.
*   **Model Registry**: Manages model versions and stages. Implements a **Champion/Challenger** strategy where the best performing model is aliased as `champion` for production use.
*   **Storage**: Uses AWS S3 for artifacts (images, models) and a SQL database (PostgreSQL/MySQL) for backend storage.

### ⚡ Prefect Pipelines (`prefect-server`, `prefect-worker`)

**Prefect** orchestrates the end-to-end MLOps workflow defined in `pipelines/flows.py`. The primary flow `test_flow` executes the following tasks:

1.  **`fetch_data`**: Downloads the raw dataset from an OCI/AWS S3 bucket and caches it locally.
2.  **`preprocess_data`**: Cleans data, computes over-ripeness metrics, filters images, and prepares the final dataset.
3.  **`upload_preprocessed_data`**: Uploads the processed dataset to **Kaggle Datasets** to be used by the training kernel.
4.  **`train_model`**:
    *   Injects hyperparameters and configuration into a **Kaggle Notebook** template.
    *   Pushes the kernel to Kaggle to run on their GPU infrastructure.
    *   Polls the kernel status until completion.
5.  **`evaluate_model`**:
    *   Compares the new model ("Challenger") against the current "Champion" in MLFlow.
    *   If the Challenger has a lower validation loss, it is promoted to `champion` in the registry.
6.  **`deploy_model`**: If a promotion occurred, this task calls the API's `/reload-model` endpoint to update the live service.

## Infrastructure & Configuration

The project uses **Docker Compose** to manage services. The configuration is split into two files for flexibility.

### 1. `docker-compose.yaml` (Production / Base)

Defines the core services for a stable, production-like environment:

*   **`postgres` & `redis`**: Backing services for Prefect.
*   **`prefect-server` & `prefect-worker`**: Runs the Prefect UI and executes flow deployments.
*   **`avocado-api`**: The production inference API.
*   **`mlflow`**: The MLFlow server connected to AWS S3 and a production DB.
*   **Traefik Labels**: Configured with labels for a Traefik reverse proxy (routing subdomains like `api.lepcodes.com`, `mlflow.lepcodes.com`).

### 2. `docker-compose.override.yaml` (Local Development)

This file overrides the base configuration for a developer-friendly setup when running `docker-compose up` locally:

*   **`avocado-api`**: Mounts the `./app` directory for **code hot-reloading**.
*   **`postgres-dev`**: A separate database container for local MLFlow development.
*   **`mlflow`**: Configured to use `postgres-dev`.
*   **`ngrok`**: Creates a public tunnel to your local MLFlow server. This is critical for **Kaggle training kernels** to report metrics back to your local machine.
*   **Ports**: Exposes services on localhost (`8080` for API, `5001` for MLFlow, `4200` for Prefect UI).

### Environment Variables

Create a `.env` file in the root directory to configure these services.

| Variable | Description | Required By |
| :--- | :--- | :--- |
| **AWS Credentials** | | |
| `AWS_ACCESS_KEY_ID` | AWS Access Key for S3/Kaggle access. | `mlflow`, `prefect` |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key. | `mlflow`, `prefect` |
| **MLFlow** | | |
| `MLFLOW_S3_ENDPOINT_URL` | Endpoint URL if using a custom S3 provider (e.g., MinIO/OCI). | `mlflow` |
| `MLFLOW_INTERNAL_URI` | Internal URI for API to reach MLFlow (e.g., `http://mlflow:5000`). | `api`, `prefect` |
| **Kaggle** | | |
| `KAGGLE_USERNAME` | Your Kaggle username for dataset upload/kernel push. | `prefect` |
| `KAGGLE_KEY` | Your Kaggle API key. | `prefect` |
| **Ngrok** | | |
| `NGROK_AUTHTOKEN` | Auth token for Ngrok (local dev only). | `ngrok` |
| **Database** | | |
| `POSTGRES_USER` | Username for local Postgres. | `postgres` |
| `POSTGRES_PASSWORD` | Password for local Postgres. | `postgres` |

## Getting Started

### Installation

To set up the project locally, you will first need to build the necessary Docker images.

#### 1. Building the Local API Development Image

This image is used for local development, enabling hot-reloading.

```bash
docker build -t avocado-ripening-api .
```

#### 2. Building and Pushing Production Base/API Image

This command builds the multi-platform base image (`lepcodes/avocado-base:latest`) that the API service uses in production environments and pushes it to a registry.

```bash
docker buildx build \
--platform linux/amd64,linux/arm64 \
-t lepcodes/avocado-base:latest \
--push .
```
*(Note: Replace `lepcodes` with your Docker Hub username.)*
*(Important: These `buildx` commands are for building and pushing your own custom images. For testing and production, pulling directly from the default `lepcodes/` sources is generally sufficient.)*

#### 3. Building and Pushing MLFlow Server Image (for custom OCI/AWS setup)

This command builds the multi-platform MLFlow server image with OCI/AWS integration and pushes it to a registry.

```bash
docker buildx build \
--platform linux/amd64,linux/arm64 \
-t lepcodes/mlflow-oci-server:latest \
-f mlflow/Dockerfile \
--push .
```
*(Note: Replace `lepcodes` with your Docker Hub username.)*
*(Important: These `buildx` commands are for building and pushing your own custom images. For testing and production, pulling directly from the default `lepcodes/` sources is generally sufficient.)*

### Running Locally

1.  **Setup Environment**: Create your `.env` file with the variables listed above.
2.  **Start Services**:
    ```bash
    docker-compose up -d
    ```
    This launches the API, MLFlow, Prefect, and DBs using the development overrides.

3.  **Access Services**:
    *   **API**: `http://localhost:8080` (Docs: `http://localhost:8080/docs`)
    *   **MLFlow UI**: `http://localhost:5000`
    *   **Prefect UI**: `http://localhost:4200`

### Running the Pipeline

To trigger the training pipeline:

1.  Access the **Prefect UI** at `http://localhost:4200`.
2.  Navigate to **Deployments**.
3.  Select **"Local Dev Deployment"** (mapped to local code) or **"Production Deployment"** (pulls from Git).
4.  Click **Run**.
