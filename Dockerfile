FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./src ./src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
