# Use the official Python image from the Docker Hub
FROM python:3.10-slim
LABEL authors="haimgoldfisher"

# Set the working directory in the container
WORKDIR /app

# Install curl, ps and other dependencies
RUN apt-get update && apt-get install -y curl  \
    && apt-get install -y procps

# Copy the requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Kafka producer and consumer scripts into the container
COPY kafka_consumer.py kafka_consumer.py

# Set the entry point to run the Kafka producer or consumer based on the argument
ENTRYPOINT ["python"]

# Default command to run the Kafka consumer
CMD ["kafka_consumer.py"]