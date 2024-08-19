#!/bin/bash

# Start Zookeeper
docker-compose up -d zookeeper
echo "Waiting for Zookeeper to be ready..."
sleep 10  # Adjust this based on how long it takes for Zookeeper to be ready

# Start Kafka
docker-compose up -d kafka
echo "Waiting for Kafka to be ready..."
sleep 10  # Adjust this as needed

# Start MongoDB
docker-compose up -d mongo
echo "Waiting for MongoDB to be ready..."
sleep 10  # Adjust this as needed

# Start Flask App
docker-compose up -d flask-app
echo "Waiting for App to be ready..."
sleep 5  # Adjust this as needed

# Start Recommendation Model
docker-compose up -d recommendation_model

# Start Kafka Consumer
docker-compose up -d kafka_consumer

echo "All services started."
