# Kafka Consumer

## Overview

This service contains a Kafka consumer microservice that listens to a Kafka topic, processes messages, and forwards them to a backend service. It uses Docker for deployment and interacts with a Kafka cluster managed with Zookeeper and the Wurstmeister Kafka Docker image.

## Features

- Consumes messages from a Kafka topic
- Sends messages to a backend service via HTTP POST requests
- Handles JSON decoding and errors
- Manually commits Kafka offsets after processing each message
- Graceful shutdown on interrupt signals

## Requirements

- Docker
- Kafka (for message production and consumption)
- Zookeeper (for Kafka coordination)
- Backend service (listening on the URL specified by `BACKEND_URL`)

## Setup

### Zookeeper and Kafka with Docker

To set up Zookeeper and Kafka using Docker, follow these steps:

1. **Create a Docker Network:**
   Create a network for your containers to communicate. This network helps in linking the Zookeeper and Kafka containers.

2. **Start Zookeeper:**
   Use the Wurstmeister Zookeeper Docker image to start a Zookeeper instance. Configure it to listen on port 2181.

3. **Start Kafka:**
   Use the Wurstmeister Kafka Docker image to start a Kafka instance. Configure it to connect to Zookeeper and listen on ports 9092 and 9093. Make sure to set the advertised listeners properly to facilitate communication between Kafka and clients.

### Building and Running the Docker Image

1. **Clone the Repository:**
   Obtain the project source code by cloning the repository from the provided URL.

2. **Create a `requirements.txt` File:**
   Include the necessary Python packages, such as `confluent_kafka` and `requests`, in this file.

3. **Build the Docker Image:**
   Build the Docker image for the Kafka consumer using the Dockerfile provided in the repository.

4. **Run the Docker Container:**
   Launch the Docker container with environment variables specifying the Kafka bootstrap servers, Kafka topic, and backend URL. Ensure the container is connected to the same Docker network as the Kafka and Zookeeper containers.

### Configuration

The Kafka consumer application requires the following environment variables:

- `KAFKA_BOOTSTRAP_SERVERS`: The address of the Kafka bootstrap servers (default: `localhost:9092`).
- `KAFKA_TOPIC`: The Kafka topic from which messages are consumed (default: `analytics_topic`).
- `BACKEND_URL`: The URL of the backend service where messages are sent (default: `http://localhost:5050/events`).

You can set these variables in your Docker container configuration.

## Usage

Once the Docker container is running, the Kafka consumer will start processing messages from the specified Kafka topic and forward them to the backend service. Logs will be output to the standard output.

## Development

To make changes to the Kafka consumer code, rebuild the Docker image after making your modifications. Ensure that you have the latest code changes reflected in the image before deploying it.

## Troubleshooting

If you encounter issues:

- Confirm that both Kafka and Zookeeper containers are running and properly connected.
- Ensure the backend service is reachable and capable of handling POST requests.
- Check the logs of the Docker containers for any errors or issues.
