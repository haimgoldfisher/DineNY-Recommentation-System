# Restaurant Recommendation System Strategy

## Problem Overview

**Problem:** Providing restaurant recommendations in New York.

**Uniqueness of the Problem:** This differs from rating and real-time location-based apps. The app is designed for frequent visitors to New York, not for users looking for immediate restaurant suggestions based on their current location. Instead, it aims to help users discover new restaurants tailored to their preferences using AI.

**Target Audience:** Frequent visitors to New York. The app does not sell products; it relies solely on user ratings within the app. There is no direct feedback on whether users followed the recommendations.

## Initial Challenges

1. **Metadata Management:** Transforming millions of records in JSON format into three collections: Users, Restaurants, and Reviews.
2. **Additional Filtering:** Based on a minimum number of reviews.
3. **Cold Start Problem:** Providing reliable recommendations without losing users during the initial phase from registration to preference setting.

## Model Training

**Algorithm Used:** ALS (Alternating Least Squares)
- **Training Frequency:** The model is trained every few hours to avoid impacting user experience. It operates as a separate entity from the backend, writing and reading to/from the database.
- **Implementation:** PySpark is used for matrix factorization due to its ability to be distributed across multiple workers, leveraging big data clusters. Training speed can be improved with more computational power.

**Additional Solution:** Clustering Algorithm
- **Purpose:** Provides quick responses to frequent user changes. After training ALS, a clustering algorithm is employed to categorize recommendations into clusters. This helps to handle frequent changes and improve response times.
- **K-Means Implementation:** K-Means with Scikit-Learn, NumPy and Pandas are used for efficient clustering operations. The clustering reduces the search space to a fixed number of clusters.

## Clustering Algorithm

**Challenges in Clustering:**

- **Randomness:** Different runs of clustering algorithms based on distance may yield varied results.
- **Distance Functions:** 
  - **Cosine Similarity:** This method calculates similarity based on vector directions, treating vectors like `[1,2,0]` and `[2,4,0]` as similar because they are proportional to each other. However, in ratings, vectors like `[1,1,1]` and `[5,5,5]` are conceptually opposite, though cosine similarity would not distinguish this difference effectively.
  - **Magnitude Difference:** To address this, I incorporated magnitude difference into the distance calculation. This approach differentiates between vectors based on their absolute magnitude. For instance, `[1,1,1]` and `[5,5,5]` are recognized as more dissimilar by considering their magnitude differences, providing a more nuanced measure of similarity.

**Solution:** 
- **Cosine Similarity with Magnitude Adjustment:** I used a custom distance function combining cosine similarity with magnitude difference. This approach calculates similarity based on both the direction and magnitude of vectors. It helps in distinguishing between vectors with different magnitudes, ensuring that vectors with similar ratings but different magnitudes are treated accordingly.

## Metrics and Evaluation

**Mathematical Metrics vs. Real-World Metrics:**
- **Mathematical Metrics:** Evaluation and RMSE (Root Mean Square Error) help assess model performance.
- **Real-World Metrics:** Key metrics include user engagement and how well the recommendations interest users. It's essential to validate that even the mathematically strongest models translate into real-world success.

## Possible Extensions

1. **Time-Based Adjustments:** Weight recent interactions more heavily.
2. **Location-Based Recommendations:** Incorporate user location for more contextual recommendations.
3. **Restaurant Categories:** Enhance user experience by categorizing restaurants.
4. **Opening Hours Filtering:** Filter recommendations based on restaurant opening times.
5. **Real-Time Updates:** Integrate live data from Google Maps for restaurant updates.

## Conclusion

The system aims to balance effective recommendation algorithms with real-time adaptability and user engagement, ensuring users receive relevant suggestions while continuously improving the model's accuracy and responsiveness.

