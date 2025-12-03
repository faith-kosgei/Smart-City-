FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY . .

# Default command to run training loop
CMD ["python", "train_prediction.py"]
