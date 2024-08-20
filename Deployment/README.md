# Project Deployment on Kubernetes

## Overview

This guide provides instructions to deploy and run your project on Kubernetes. The project includes the following components:

- **Zookeeper**: Used for managing Kafka brokers.
- **Kafka**: Messaging system.
- **MongoDB**: Database for storing data.
- **Flask Application**: Web application.
- **Recommendation Model**: Generates recommendations.
- **Kafka Consumer**: Processes Kafka messages.

## Prerequisites

- **Kubernetes Cluster**: Ensure you have a Kubernetes cluster set up (e.g., Minikube or a cloud-based Kubernetes service).
- **kubectl**: Command-line tool for Kubernetes.
- **Docker**: Images for your services should be available, or use public images.

## Deployment Steps

### 1. Start Minikube (if using Minikube):

Run the following command to start Minikube:

    minikube start

### 2. Apply Zookeeper Deployment and Service:

Run the following commands to apply the Zookeeper deployment and service:

    kubectl apply -f zookeeper-deployment.yaml
    kubectl apply -f zookeeper-service.yaml

Wait for Zookeeper to be ready:

    kubectl wait --for=condition=available --timeout=120s deployment/zookeeper

### 3. Apply Kafka Deployment and Service:

Run the following commands to apply the Kafka deployment and service:

    kubectl apply -f kafka-deployment.yaml
    kubectl apply -f kafka-service.yaml

Wait for Kafka to be ready:

    kubectl wait --for=condition=available --timeout=120s deployment/kafka

### 4. Apply MongoDB Deployment and Service:

Run the following commands to apply the MongoDB deployment and service:

    kubectl apply -f mongo-deployment.yaml
    kubectl apply -f mongo-service.yaml

Wait for MongoDB to be ready:

    kubectl wait --for=condition=available --timeout=120s deployment/mongo

### 5. Apply Flask Application Deployment and Service:

Run the following commands to apply the Flask application deployment and service:

    kubectl apply -f flask-app-deployment.yaml
    kubectl apply -f flask-app-service.yaml

Wait for the Flask application to be ready:

    kubectl wait --for=condition=available --timeout=120s deployment/flask-app

### 6. Apply Recommendation Model Deployment and Service:

Run the following commands to apply the recommendation model deployment and service:

    kubectl apply -f recommendation-model-deployment.yaml
    kubectl apply -f recommendation-model-service.yaml

Wait for the recommendation model to be ready:

    kubectl wait --for=condition=available --timeout=120s deployment/recommendation-model

### 7. Apply Kafka Consumer Deployment:

Run the following command to apply the Kafka consumer deployment:

    kubectl apply -f kafka-consumer-deployment.yaml

Wait for the Kafka consumer to be ready:

    kubectl wait --for=condition=available --timeout=120s deployment/kafka-consumer

## Accessing the Flask Application

To access the Flask application service, run:

    minikube service flask-app

This will open a browser window with the URL to access the Flask application.

## Troubleshooting

- **CrashLoopBackOff Errors**: Check the logs of the failing pods using `kubectl logs <pod-name>`. Review the error messages to debug configuration issues.
- **Service Connectivity**: Ensure all services can communicate with each other. Use `kubectl exec` to run commands inside pods for debugging.

## Cleanup

To delete all resources created for this project, run:

    kubectl delete -f zookeeper-deployment.yaml
    kubectl delete -f zookeeper-service.yaml
    kubectl delete -f kafka-deployment.yaml
    kubectl delete -f kafka-service.yaml
    kubectl delete -f mongo-deployment.yaml
    kubectl delete -f mongo-service.yaml
    kubectl delete -f flask-app-deployment.yaml
    kubectl delete -f flask-app-service.yaml
    kubectl delete -f recommendation-model-deployment.yaml
    kubectl delete -f recommendation-model-service.yaml
    kubectl delete -f kafka-consumer-deployment.yaml

## Notes

- Ensure all YAML files are correctly configured before applying them.
- Adjust timeout values as needed based on your cluster performance and service startup times.
- Monitor the status of your deployments using `kubectl get deployments` and `kubectl get pods` to ensure everything is running as expected.
min