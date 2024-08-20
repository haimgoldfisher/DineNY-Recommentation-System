# DineNY - A Restaurant Recommendation System in New York

<img src="App/static/imgs/animation.gif" alt="Animated GIF" width="600">


By Haim Goldfisher

## Background - Restaurant Recommendation System Strategy

### Problem Overview

**Problem:** Providing restaurant recommendations in New York.

**Uniqueness of the Problem:** Unlike rating and real-time location-based apps, this system is designed for frequent visitors to New York. It helps users discover new restaurants tailored to their preferences using AI, rather than suggesting immediate options based on current location. 

**Target Audience:** Frequent visitors to New York. The app relies solely on user ratings within the app and does not sell products or provide direct feedback on whether users followed the recommendations.

### Initial Challenges

1. **Metadata Management:** Transforming millions of records in JSON format into three collections: Users, Restaurants, and Reviews.
2. **Additional Filtering:** Implementing filtering based on a minimum number of reviews.
3. **Cold Start Problem:** Addressing the challenge of providing reliable recommendations from the initial phase of registration to preference setting without losing users.

### Model Training

**Algorithm Used:** ALS (Alternating Least Squares)
- **Training Frequency:** The model is trained every few hours to minimize impact on user experience. It operates as a separate entity from the backend, managing database read and write operations independently.
- **Implementation:** PySpark is employed for matrix factorization, leveraging distributed computing on big data clusters to improve training speed.

**Additional Solution:** Clustering Algorithm
- **Purpose:** To handle frequent user changes efficiently. After ALS training, a clustering algorithm is used to categorize recommendations into clusters, improving response times.
- **K-Means Implementation:** Utilizes Scikit-Learn, NumPy, and Pandas for efficient clustering. The algorithm reduces the search space to a fixed number of clusters.

### Clustering Algorithm

**Challenges in Clustering:**

- **Randomness:** Different runs of clustering algorithms may yield varied results.
- **Distance Functions:**
  - **Cosine Similarity:** Measures similarity based on vector directions. However, it may not effectively distinguish between vectors with different magnitudes.
  - **Magnitude Difference:** To address this, magnitude difference is incorporated into the distance calculation, distinguishing between vectors based on their absolute magnitude.

**Solution:**
- **Cosine Similarity with Magnitude Adjustment:** A custom distance function combining cosine similarity with magnitude difference is used. This approach calculates similarity based on both vector direction and magnitude, providing a more nuanced measure of similarity.

### Metrics and Evaluation

**Mathematical Metrics vs. Real-World Metrics:**
- **Mathematical Metrics:** RMSE (Root Mean Square Error) is used to assess model performance.
- **Real-World Metrics:** Key metrics include user engagement and the relevance of recommendations. It's crucial to ensure that mathematically strong models translate into real-world success.

### Possible Extensions

1. **Time-Based Adjustments:** Weight recent interactions more heavily.
2. **Location-Based Recommendations:** Incorporate user location for contextual recommendations.
3. **Restaurant Categories:** Enhance user experience by categorizing restaurants.
4. **Opening Hours Filtering:** Filter recommendations based on restaurant opening times.
5. **Real-Time Updates:** Integrate live data from Google Maps for up-to-date restaurant information.


## Project Overview

### App Microservice

The App microservice is responsible for the main Flask application that provides the user interface and handles interactions with the recommendation system.

For more details, see [App README](App/README.md).

### MongoDB Microservice

The MongoDB microservice manages the database that stores user data, restaurant information, and reviews.

For more details, see [DB README](DB/README.md).

### AI Recommendation Model Microservice

The Model microservice handles the training and execution of the recommendation algorithms, including ALS and clustering models.

For more details, see [Model README](Model/README.md).

### Data-Streaming Microservice

The Data-Streaming microservice handles real-time data streaming and processing using Kafka.

For more details, see [Data-Streaming README](Data-Streaming/README.md).

### Deployment

The Deployment section contains configurations for deploying the services using Kubernetes, including setup for Kafka, MongoDB, and the Flask app.

For more details, see [Deployment README](Deployment/README.md).

### File Structure

<details>
<summary>App</summary>

  - `Dockerfile`: Dockerfile for the Flask application.
  - `__pycache__`: Compiled Python files.
  - `clustering_utils.py`: Utilities for clustering operations.
  - `requirements.txt`: Python dependencies for the Flask application.
  - `templates`: HTML templates for the Flask application.
  - `app.py`: Main Flask application script.
  - `get_cluster.py`: Script for generating clusters.
  - `static`: Static files such as CSS and JavaScript.

For more details, see [App README](App/README.md).

</details>

<details>
<summary>DB</summary>

  - `Dockerfile`: Dockerfile for the database setup.
  - `README.md`: Documentation for the database.
  - `data-cleaning`: Scripts for data cleaning.
  - `dump`: Database dump files.
  - `get_img.py`: Script for retrieving images.
  - `restore.sh`: Script for restoring the database.

For more details, see [DB README](DB/README.md).

</details>

<details>
<summary>Model</summary>

  - `Dockerfile`: Dockerfile for the model training environment.
  - `__pycache__`: Compiled Python files.
  - `clus_counter.py`: Script for counting clusters.
  - `clustering_training.py`: Script for training the clustering model.
  - `model.py`: Script for the ALS model.
  - `README.md`: Documentation for the model training.
  - `als_training.py`: Script for training the ALS model.
  - `clus_test.py`: Script for testing clustering.
  - `dists.py`: Script for distance functions used in clustering.
  - `requirements.txt`: Python dependencies for model training.

For more details, see [Model README](Model/README.md).

</details>

<details>
<summary>Data-Streaming</summary>

  - `Dockerfile`: Dockerfile for the data streaming service.
  - `README.md`: Documentation for the data streaming service.
  - `kafka_consumer.py`: Kafka consumer script.
  - `requirements.txt`: Python dependencies for the data streaming service.

For more details, see [Data-Streaming README](Data-Streaming/README.md).

</details>

<details>
<summary>Deployment</summary>

  - `README.md`: Documentation for deployment.
  - `deploy.sh`: Deployment script.
  - `kafka-consumer-deployment.yaml`: Kubernetes deployment configuration for Kafka consumer.
  - `mongo-service.yaml`: Kubernetes service configuration for MongoDB.
  - `kafka-deployment.yaml`: Kubernetes deployment configuration for Kafka.
  - `recommendation-model-deployment.yaml`: Kubernetes deployment configuration for the recommendation model.
  - `flask-app-deployment.yaml`: Kubernetes deployment configuration for the Flask app.
  - `kafka-service.yaml`: Kubernetes service configuration for Kafka.
  - `recommendation-model-service.yaml`: Kubernetes service configuration for the recommendation model.
  - `flask-app-service.yaml`: Kubernetes service configuration for the Flask app.
  - `mongo-deployment.yaml`: Kubernetes deployment configuration for MongoDB.
  - `zookeeper-deployment.yaml`: Kubernetes deployment configuration for Zookeeper.
  - `zookeeper-service.yaml`: Kubernetes service configuration for Zookeeper.

For more details, see [Deployment README](Deployment/README.md).

</details>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
