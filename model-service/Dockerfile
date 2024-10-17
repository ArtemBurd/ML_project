FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY saved_models ./saved_models
COPY model_class.py model-service.py ./
CMD ["python", "model-service.py"]