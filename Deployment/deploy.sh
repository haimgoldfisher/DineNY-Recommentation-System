#!/bin/bash

# Apply Zookeeper Deployment and Service
echo "Applying Zookeeper Deployment and Service..."
kubectl apply -f zookeeper-deployment.yaml
kubectl apply -f zookeeper-service.yaml
echo "Zookeeper Deployment and Service applied."

# Wait for Zookeeper to be ready
echo "Waiting for Zookeeper to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/zookeeper

# Apply Kafka Deployment and Service
echo "Applying Kafka Deployment and Service..."
kubectl apply -f kafka-deployment.yaml
kubectl apply -f kafka-service.yaml
echo "Kafka Deployment and Service applied."

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/kafka

# Apply MongoDB Deployment and Service
echo "Applying MongoDB Deployment and Service..."
kubectl apply -f mongo-deployment.yaml
kubectl apply -f mongo-service.yaml
echo "MongoDB Deployment and Service applied."

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/mongo

# Apply Flask App Deployment and Service
echo "Applying Flask App Deployment and Service..."
kubectl apply -f flask-app-deployment.yaml
kubectl apply -f flask-app-service.yaml
echo "Flask App Deployment and Service applied."

# Wait for Flask App to be ready
echo "Waiting for Flask App to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/flask-app

# Apply Recommendation Model Deployment and Service
echo "Applying Recommendation Model Deployment and Service..."
kubectl apply -f recommendation-model-deployment.yaml
kubectl apply -f recommendation-model-service.yaml
echo "Recommendation Model Deployment and Service applied."

# Wait for Recommendation Model to be ready
echo "Waiting for Recommendation Model to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/recommendation-model

# Apply Kafka Consumer Deployment
echo "Applying Kafka Consumer Deployment..."
kubectl apply -f kafka-consumer-deployment.yaml
echo "Kafka Consumer Deployment applied."

# Wait for Kafka Consumer to be ready
echo "Waiting for Kafka Consumer to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/kafka-consumer

echo "All deployments applied successfully."
