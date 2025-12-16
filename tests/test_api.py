from unittest.mock import MagicMock


def test_health_check(client):
    """
    Verifies that the model was loaded into memory during startup
    and the /health endpoint can use it.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Model is healthy"}


def test_predict_upload_file(client):
    """
    Happy Path: Uploading a file.
    """
    # Create a dummy image
    import io

    from PIL import Image

    img = Image.new("RGB", (100, 100), color="green")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")

    files = {"image_file": ("test.png", img_byte_arr.getvalue(), "image/png")}
    data = {"storage_condition": "T10"}

    response = client.post("/predict", files=files, data=data)

    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["prediction"][0]["estimated_days"] == 3.5


def test_predict_image_url(client):
    """
    Happy Path: Providing a URL.
    This relies on the `requests.get` patch in conftest.py
    """
    data = {
        "image_url": "http://example.com/avocado.png",
        "image_name": "avocado.png",
        "storage_condition": "Tam",
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 200
    assert response.json()["prediction"][0]["estimated_days"] == 3.5


def test_validation_no_image(client):
    """
    Sad Path: Logic validation (FastAPI handles this before hitting the model).
    """
    data = {"storage_condition": "Tam"}
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert "Image data missing" in response.json()["detail"]


def test_model_inference_crash(client, mock_external_deps):
    """
    Sad Path: The model is loaded, but .predict() fails internally.
    We override the mock behavior just for this test.
    """
    # Force the mock model to raise an exception
    mock_external_deps.predict = MagicMock(side_effect=Exception("GPU OOM Error"))

    # Prepare valid input
    import io

    from PIL import Image

    img = Image.new("RGB", (1, 1), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    files = {"image_file": ("test.png", img_byte_arr.getvalue(), "image/png")}

    response = client.post("/predict", files=files)

    assert response.status_code == 500
    assert "GPU OOM Error" in response.json()["detail"]  # Expect 500
    assert response.status_code == 500
    assert "GPU OOM Error" in response.json()["detail"]
