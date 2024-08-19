# Use a base image with Python and necessary tools
FROM bitnami/spark:3.4.1
LABEL authors="haimgoldfisher"

# Switch to root user to install dependencies
USER root

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set up working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8080

# Run the FastAPI server
CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
