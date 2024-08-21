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

### Dataset

#### Overview

The dataset for this project includes information about restaurants, user reviews, and user data. It consists of the following primary components:

1. **Restaurants Data:** Details about restaurants, including names, locations, and other metadata.
2. **User Data:** Information about users, including their preferences.
3. **Reviews Data:** User-generated reviews with ratings, comments, and timestamps.

#### Sources

- **Google Local Dataset (2021):** This dataset provides detailed information about Google Maps places. You can access the dataset [Here](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/).
- **User Contributions:** Data from user interactions within the application is continuously added to refine recommendations and improve accuracy.

#### Data Processing

1. **Data Cleaning:** Raw data is cleaned and normalized to ensure consistency and accuracy.
2. **Data Transformation:** The data is transformed into a suitable format for analysis and model training.
3. **Integration:** Data is integrated into the MongoDB database, structured into collections for efficient querying and retrieval.

#### Data Schema

- **Restaurants Collection:** Includes fields such as `name`, `address`, `gmap_url`, and more.
- **Users Collection:** Includes fields such as `user_id`, and `reviews` array.
- **Reviews Collection:** Includes fields such as `user_id`, `gmap_id`, `rating`, and `timestamp`.


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

  - `app.py`: Main Flask application script.
  - `get_cluster.py`: Script for return user's cluster.
  - `clustering_utils.py`: Utilities for clustering operations.
  - `static`: Static files such as CSS and JavaScript.
  - `templates`: HTML templates for the Flask application.
  - `README.md`: Documentation for the application.
  - `requirements.txt`: Python dependencies for the Flask application.
  - `Dockerfile`: Dockerfile for the Flask application.

</details>

<details>
<summary>DB</summary>

  - `data-cleaning`: Scripts for data cleaning.
  - `dump`: Database dump files.
  - `get_img.py`: Script for retrieving images.
  - `restore.sh`: Script for restoring the database in MongoDB container.
  - `README.md`: Documentation for the database.
  - `Dockerfile`: Dockerfile for the database setup.

</details>

<details>
<summary>Model</summary>

  - `model.py`: Script for the FAST API - training scheduler.
  - `als_training.py`: Script for training the ALS model.
  - `clustering_training.py`: Script for training the clustering model.
  - `dists.py`: Script for distance functions used in clustering.
  - `clus_test.py`: Script for testing clustering.
  - `clus_counter.py`: Script for counting clusters.
  - `README.md`: Documentation for the model training.
  - `requirements.txt`: Python dependencies for model training.
  - `Dockerfile`: Dockerfile for the model training environment.

</details>

<details>
<summary>Data-Streaming</summary>

  - `kafka_consumer.py`: Kafka consumer script.
  - `README.md`: Documentation for the data streaming service.
  - `requirements.txt`: Python dependencies for the data streaming service.
  - `Dockerfile`: Dockerfile for the data streaming service.

</details>

<details>
<summary>Deployment</summary>

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
  - `README.md`: Documentation for deployment.

</details>

## Technologies

<div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">

  <a href="https://www.python.org/" target="_blank" title="Python">
    <img src="https://www.svgrepo.com/show/376344/python.svg" alt="Python" width="80"/>
  </a>

  <a href="https://developer.mozilla.org/en-US/docs/Web/HTML" target="_blank" title="HTML">
    <img src="https://cdn.iconscout.com/icon/premium/png-256-thumb/html-2752158-2284975.png?f=webp&w=256" alt="HTML" width="80"/>
  </a>

  <a href="https://developer.mozilla.org/en-US/docs/Web/CSS" target="_blank" title="CSS">
    <img src="https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/css-icon.png" alt="CSS" width="80"/>
  </a>

  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" title="JavaScript">
    <img src="https://cdn.worldvectorlogo.com/logos/javascript-1.svg" alt="JavaScript" width="80"/>
  </a>

  <a href="https://flask.palletsprojects.com/" target="_blank" title="Flask">
    <img src="https://external-preview.redd.it/n9EWl-GXdiaYYVOhB3Dy1hT69l0v8KfPnDVeqDQ6ANE.jpg?width=640&crop=smart&auto=webp&s=2d2869322e0dc4ca537a9b71295e4e9f1b3e9a58" alt="Flask" width="80"/>
  </a>

  <a href="https://www.mongodb.com/" target="_blank" title="MongoDB">
    <img src="https://www.svgrepo.com/show/331488/mongodb.svg" alt="MongoDB" width="80"/>
  </a>

  <a href="https://www.uvicorn.org/" target="_blank" title="Uvicorn">
    <img src="https://www.uvicorn.org/uvicorn.png" alt="Uvicorn" width="80"/>
  </a>

  <a href="https://scikit-learn.org/" target="_blank" title="Scikit-Learn">
    <img src="https://seeklogo.com/images/S/scikit-learn-logo-8766D07E2E-seeklogo.com.png" alt="scikit-learn" width="80"/>
  </a>

  <a href="https://pandas.pydata.org/" target="_blank" title="Pandas">
    <img src="https://pandas.pydata.org/docs/_static/pandas.svg" alt="Pandas" width="80"/>
  </a>

  <a href="https://numpy.org/" target="_blank" title="NumPy">
    <img src="https://numpy.org/images/logo.svg" alt="NumPy" width="80"/>
  </a>

  <a href="https://scipy.org/" target="_blank" title="SciPy">
    <img src="https://miro.medium.com/v2/resize:fit:300/1*QfoEbdmoC1fJcRSnaQmtRg.png" alt="SciPy" width="80"/>
  </a>

  <a href="https://spark.apache.org/" target="_blank" title="PySpark">
    <img src="https://spark.apache.org/images/spark-logo-trademark.png" alt="PySpark" width="80"/>
  </a>

  <a href="https://kafka.apache.org/" target="_blank" title="Kafka">
    <img src="https://cdn.prod.website-files.com/62038ffc9cd2db4558e3c7b7/623b44a1913c46041e39c836_kafka.svg" alt="Kafka" width="80"/>
  </a>

  <a href="https://www.docker.com/" target="_blank" title="Docker">
    <img src="https://www.svgrepo.com/show/331370/docker.svg" alt="Docker" width="80"/>
  </a>

  <a href="https://kubernetes.io/" target="_blank" title="Kubernetes">
    <img src="https://logos-world.net/wp-content/uploads/2023/06/Kubernetes-Symbol.png" alt="Kubernetes" width="80"/>
  </a>

  <a href="https://minikube.sigs.k8s.io/" target="_blank" title="Minikube">
    <img src="https://cdn.prod.website-files.com/64196dbe03e13c204de1b1c8/64773f546a9ff7246f6a73f0_80-image-2.png" alt="Minikube" width="80"/>
  </a>

</div>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
