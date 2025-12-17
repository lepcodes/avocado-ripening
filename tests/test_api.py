import io
from unittest.mock import MagicMock, patch

from PIL import Image


def test_health_check(client):
    """
    Verifies that the model was loaded into memory during startup
    and the /health endpoint can use it.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Model is healthy"}


def create_dummy_image_bytes(color="green"):
    img = Image.new("RGB", (100, 100), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def test_predict_upload_file(client):
    """
    Happy Path: Uploading a file.
    """
    img_bytes = create_dummy_image_bytes("green")
    files = [("image_files", ("test.png", img_bytes, "image/png"))]
    data = {"storage_conditions": ["T10"]}

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
        "image_urls": ["http://example.com/avocado.png"],
        "storage_conditions": ["Tam"],
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 200
    assert response.json()["prediction"][0]["estimated_days"] == 3.5


def test_validation_no_image(client):
    """
    Sad Path: Logic validation (FastAPI handles this before hitting the model).
    """
    data = {"storage_conditions": ["Tam"]}
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert "No images provided" in response.json()["detail"]


def test_model_inference_crash(client, mock_external_deps):
    """
    Sad Path: The model is loaded, but .predict() fails internally.
    We override the mock behavior just for this test.
    """
    # Force the mock model to raise an exception
    mock_external_deps.predict = MagicMock(side_effect=Exception("GPU OOM Error"))

    # Prepare valid input
    img_bytes = create_dummy_image_bytes("red")
    files = [("image_files", ("test.png", img_bytes, "image/png"))]
    data = {"storage_conditions": ["T10"]}

    response = client.post("/predict", files=files, data=data)

    assert response.status_code == 500
    assert "GPU OOM Error" in response.json()["detail"]  # Expect 500


def test_predict_multiple_files(client):
    """
    Happy Path: Uploading multiple files with matching storage conditions.
    """
    img1 = create_dummy_image_bytes("green")
    img2 = create_dummy_image_bytes("red")

    files = [
        ("image_files", ("img1.png", img1, "image/png")),
        ("image_files", ("img2.png", img2, "image/png")),
    ]
    data = {"storage_conditions": ["T10", "T20"]}

    response = client.post("/predict", files=files, data=data)

    assert response.status_code == 200
    json_resp = response.json()
    assert len(json_resp["prediction"]) == 2


def test_predict_multiple_urls(client):
    """
    Happy Path: Providing multiple URLs.
    """
    img_bytes = create_dummy_image_bytes("blue")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = img_bytes

    with patch("app.main.requests.get", return_value=mock_response):
        data = {
            "image_urls": ["http://site.com/1.png", "http://site.com/2.png"],
            "storage_conditions": ["Tam", "T10"],
        }

        response = client.post("/predict", data=data)

        assert response.status_code == 200
        assert len(response.json()["prediction"]) == 2


def test_predict_mixed_files_and_urls(client):
    """
    Happy Path: 1 File + 1 URL.
    """
    img_file = create_dummy_image_bytes("yellow")
    img_url_bytes = create_dummy_image_bytes("purple")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = img_url_bytes

    with patch("app.main.requests.get", return_value=mock_response):
        files = [("image_files", ("file1.png", img_file, "image/png"))]
        data = {
            "image_urls": ["http://site.com/url1.png"],
            "storage_conditions": ["T10", "Tam"],
        }

        response = client.post("/predict", files=files, data=data)

        assert response.status_code == 200
        assert len(response.json()["prediction"]) == 2


def test_mismatch_counts(client):
    """
    Sad Path: 2 Images (files) but 1 Storage Condition.
    """
    img1 = create_dummy_image_bytes("green")
    img2 = create_dummy_image_bytes("green")

    files = [
        ("image_files", ("img1.png", img1, "image/png")),
        ("image_files", ("img2.png", img2, "image/png")),
    ]
    data = {"storage_conditions": ["T10"]}

    response = client.post("/predict", files=files, data=data)

    assert response.status_code == 400
    assert (
        "Number of storage conditions and images do not match"
        in response.json()["detail"]
    )


def test_url_failure_in_batch(client):
    """
    Sad Path: 2 URLs, one fails.
    """
    img_bytes = create_dummy_image_bytes("white")
    valid_response = MagicMock()
    valid_response.status_code = 200
    valid_response.content = img_bytes

    # requests.get will be called twice. First succeeds, second raises exception.
    with patch(
        "app.main.requests.get",
        side_effect=[valid_response, Exception("Connection Timeout")],
    ):
        data = {
            "image_urls": ["http://good.com/img.png", "http://bad.com/img.png"],
            "storage_conditions": ["T10", "T10"],
        }

        response = client.post("/predict", data=data)

        assert response.status_code == 400
        assert "Error loading image from URL" in response.json()["detail"]
        assert "Error loading image from URL" in response.json()["detail"]
