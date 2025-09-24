# avocado-ripening  🥑

A Deep Learning project for predicting the ripening of avocado images.

## Building the Docker Image

To build the Docker image for development, run the following command:

```bash
docker build -t avocado-ripening-api .
```

and then run the container with the following command:

## Running the Docker Container

To run the Docker container, use the following command:

```bash
docker run -p 8080:80 avocado-ripening
```

This will start the container and map port 80 of the container to port 8080 of the host machine.

You can also create a docker-compose.yaml file with the following content to run the container locally:

```yaml
services:
    api:
        container_name: avocado-api
        build: .
        command: uvicorn src.main:app --host 0.0.0.0 --port 80 --reload
        ports:
        - "8080:80"
        volumes:
        - ./models:/app/models
        - ./src:/app/src
        environment:
        - MODEL_NAME={your_model_name}.keras
        - MODEL_DIR=/app/models
        - TF_ENABLE_ONEDNN_OPTS=0
        - TF_CPP_MIN_LOG_LEVEL=2
        restart: unless-stopped
```

To build the Docker image for production, run the following command (you need to replace lepcodes with your Docker Hub username):

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t lepcodes/avocado-ripening-api:latest --push .
```

# MLFlow Server

The MLFlow server is a lightweight, open source platform for managing the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

## Running the OCI MLFlow Server using Docker Compose

Create a docker-compose.yaml file with the following content to run the server locally:

```yaml
services:
    mlflow:
        container_name: mlflow-server
        build:
        context: ./mlflow
        command: >
            mlflow server
            --host 0.0.0.0
            --port 5000
            --backend-store-uri file:///mlruns/backend
            --default-artifact-root ./mlruns/artifacts
        ports:
        - "5001:5000"
        restart: unless-stopped
```
or use the published image and use a remote bucket and a backend store:

```yaml
services:
    mlflow:
        container_name: mlflow-server
        image: lepcodes/mlflow-oci-server:latest
        command: >
            mlflow server
            --host 0.0.0.0
            --port 5000
            --backend-store-uri mysql+pymysql://{username}:{password}@{host}:{port}/{database}
            --default-artifact-root s3://{bucket_name}/
        ports:
        - "5001:5000"
        env_file:
        - .env
        restart: unless-stopped
```
You need to create a .env file with the following content:

```bash
AWS_ACCESS_KEY_ID={your_access_key_id}
AWS_SECRET_ACCESS_KEY={your_secret_access_key}
MLFLOW_S3_ENDPOINT_URL={your_s3_endpoint_url}
``` 

And then run the container with the following command:

```bash
docker-compose up -d
```

This will start the container and map port 5000 of the container to port 5001 of the host machine.

## Building the Docker Image for pushing to Docker Hub

```bash
docker buildx build \
--platform linux/amd64,linux/arm64 \
-t lepcodes/mlflow-oci-server:latest \
-f mlflow/Dockerfile \
--push .
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         avocado_ripening and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes avocado_ripening a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

