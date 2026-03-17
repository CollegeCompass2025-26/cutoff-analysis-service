# Use a slim version of Python as a base
FROM python:3.11-slim

# Install system dependencies for ML libraries (XGBoost, psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
# Note: TensorFlow is huge. Using --no-cache-dir to save space.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set PYTHONPATH to ensure internal modules are findable
ENV PYTHONPATH=/app

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the application using uvicorn
# We use 0.0.0.0 to allow external traffic in Render
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
