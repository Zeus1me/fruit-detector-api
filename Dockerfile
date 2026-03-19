# Use a slim Python 3.11 base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer if unchanged)
COPY requirements.txt .

# Install dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 8000 for the FastAPI server
EXPOSE 7860

# Start the server with uvicorn
# host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
