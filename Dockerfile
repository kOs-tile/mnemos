FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download sentence transformer model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY mnemos/ ./mnemos/

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "mnemos.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
