# Restaurant Recommender Model Service

## Overview

This service provides an end-to-end pipeline for generating and managing restaurant recommendations using a combination of Alternating Least Squares (ALS) model and K-Means clustering. The service is composed of three main components:

1. **ALS Model Training**: Generates personalized restaurant recommendations based on user ratings.
2. **Clustering**: Groups users into clusters based on their rating behavior and generates top restaurant recommendations for each cluster.
3. **API Endpoint**: Provides a REST API to trigger model training and check the next scheduled training time.

## Components

### 1. ALS Model Training

The ALS model is trained using user ratings data from MongoDB. The model generates personalized restaurant recommendations for each user, which are then saved back to MongoDB.

#### Key Steps:
- **Data Loading**: Loads review data from MongoDB.
- **Preprocessing**: Indexes user and restaurant IDs and filters relevant columns.
- **Training**: Trains the ALS model with the preprocessed data.
- **Evaluation**: Evaluates the model performance using RMSE (Root Mean Square Error).
- **Recommendation Generation**: Generates top N recommendations for each user.
- **Saving**: Saves the generated recommendations to MongoDB.

#### ALS Model Configuration (after a grid search with cross validation):
- **Max Iterations**: 10
- **Regularization Parameter**: 0.1
- **Cold Start Strategy**: Drop (for users with no history)
- **Spark Configuration**:
  - Executor Memory: 8g
  - Driver Memory: 8g
  - Executor Cores: 4
  - Shuffle Partitions: 400

### 2. Clustering

Clustering is performed on the user-rating matrix to group similar users together. K-Means clustering with a custom distance metric is used to create clusters, and top restaurant recommendations are generated for each cluster.

#### Key Steps:
- **Data Loading**: Fetches user ratings from MongoDB.
- **Preprocessing**: Filters out users without history, then pivots the data into a user-rating matrix.
- **Clustering**: Performs K-Means clustering on the user-rating matrix.
- **Recommendation Generation**: Generates top N restaurant recommendations for each cluster.
- **Saving**: Saves the cluster recommendations to each cluster in MongoDB.

#### Clustering Configuration:
- **Number of Clusters (k)**: 50 (due to elbow method results).
- **Distance Metric**: Cosine Similarity with Magnitude.

### 3. API Endpoint

A FastAPI-based REST API that allows triggering model training and retrieving the next scheduled training time.

#### Endpoints:
- **POST /train**: Triggers ALS model training in the background.
- **GET /next-training**: Retrieves the time remaining until the next scheduled training.

#### Background Scheduling:
- The models training is automatically scheduled to run every 3 hours.

### Notes

- The clustering training script (`clustering_training.py`) is automatically invoked after ALS training completes.
- The ALS results are written directly into `Recommendations` collection while K-Means results are written into `Clusters` collection. 
- The clustering model params are written each training cycle into `Metadata` collection in order to let the backend use them.
- Make sure that sufficient memory is allocated to the Spark executors to handle large datasets.
- Adjust the number of users and restaurants in clustering as per your dataset size and computational capacity.

## File Structure

- `als_training.py`: Script for ALS model training.
- `clustering_training.py`: Script for running K-Means clustering.
- `dists.py`: Util functions for K-Means clustering.
- `clus_test.py`: Tester for the K-Means clustering pre-trained model.
- `clus_counter.py`: Sanity check function that maps the users into their clusters.
- `model.py`: FastAPI service to manage and schedule training.
- `requirements.txt`: List of Python dependencies.
- `Dockerfile`: For deployment.
