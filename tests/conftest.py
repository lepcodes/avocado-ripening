import os
from unittest.mock import MagicMock, patch

import mlflow.tensorflow
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

os.environ["MLFLOW_INTERNAL_URI"] = "http://mock-server"


class MockKerasModel:
    """
    Acts exactly like your loaded TF model.
    It has a predict method that accepts a dictionary and returns a numpy array.
    """

    def predict(self, inputs):
        # inputs is a dict: {'image_input': ..., 'condition_input': ...}
        # Return shape (batch_size, 1) to mimic the prediction output
        batch_size = len(inputs["image_input"])
        return np.full((batch_size, 1), 3.5, dtype=np.float32)

    def save(self, filepath):
        pass


# 3. The Fixture that applies the patches
@pytest.fixture(scope="function")
def mock_external_deps():
    """
    Patches MLflow and Requests.
    Returns the mock model instance so we can modify it in tests (e.g., make it raise errors).
    """
    mock_model_instance = MockKerasModel()

    with patch.object(
        mlflow.tensorflow, "load_model", return_value=mock_model_instance
    ):
        with patch("mlflow.set_tracking_uri"):
            with patch.object(requests, "get") as mock_requests:
                # Setup a fake response for requests
                mock_response = MagicMock()
                mock_response.status_code = 200
                # Create a tiny 1x1 red pixel image
                import io

                from PIL import Image

                img = Image.new("RGB", (1, 1), color="red")
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                mock_response.content = img_byte_arr.getvalue()

                mock_requests.return_value = mock_response

                yield mock_model_instance


@pytest.fixture(scope="function")
def client(mock_external_deps):
    """
    Creates the client. Crucially, 'mock_external_deps' is passed in,
    ensuring mocks are active BEFORE the app starts up and runs lifespan.
    """
    from app.main import app

    # TestClient(app) triggers the lifespan (startup) event
    with TestClient(app) as client:
        yield client
