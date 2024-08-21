import numpy as np
import pandas as pd
import joblib
from bson import ObjectId
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from dists import kmeans_clustering_plus_plus, calculate_silhouette_score, cos_sim_mag_dist
import os

def run_clustering(k=50):
    # MongoDB connection details
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(mongo_uri)
    db = client['Google-Maps-Restaurant']
    reviews_collection = db["Reviews"]
    clusters_collection = db['Clusters']
    metadata_collection = db['Metadata']

    # Fetch all reviews from MongoDB
    reviews_cursor = reviews_collection.find({}, {'_id': 0, 'user_id': 1, 'gmap_id': 1, 'rating': 1})
    reviews = pd.DataFrame(list(reviews_cursor))

    # Handle duplicates by aggregating (e.g., taking the average rating for each (user_id, gmap_id) pair)
    reviews = reviews.groupby(['user_id', 'gmap_id']).agg({'rating': 'mean'}).reset_index()

    # Count the number of reviews per user
    user_review_counts = reviews['user_id'].value_counts()

    # Sort users by the number of reviews and select the top N users (e.g., top 150,000 users)
    top_users = user_review_counts.head(150000).index  # Adjust the number here as needed

    # Filter reviews to include only the selected top users
    filtered_reviews = reviews[reviews['user_id'].isin(top_users)]

    # Pivot the DB: user_id as rows, gmap_id as columns, rating as values
    users = filtered_reviews.pivot(index='user_id', columns='gmap_id', values='rating').fillna(0)

    print('users.shape before filtering:', users.shape)

    # Sort users by the number of nonzero columns (i.e., the number of restaurants reviewed)
    users = users.reindex(users.astype(bool).sum(axis=1).sort_values(ascending=False).index)

    # Select the top 100,000 users (or change this to a different number if needed)
    users = users[:100000]
    print('users.shape after selecting top users:', users.shape)

    # Remove restaurants reviewed by fewer than a certain number of people (e.g., 30)
    users = users.loc[:, users.astype(bool).sum(axis=0) >= 30]
    print('users.shape after filtering restaurants:', users.shape)

    # Save the feature columns to MongoDB
    metadata_collection.replace_one(
        {'_id': 'train_columns'},
        {'_id': 'train_columns', 'columns': users.columns.tolist()},
        upsert=True
    )

    # Check if there are still enough columns to continue
    if users.shape[1] == 0:
        raise ValueError("No restaurants left after filtering. Consider lowering the threshold.")

    # Train-test split: 90% for training, 10% for testing
    train = users.iloc[:int(users.shape[0] * 0.9), :]
    test = users.iloc[int(users.shape[0] * 0.9):, :]
    print('train.shape:', train.shape)
    print('test.shape:', test.shape)

    # Convert the training DB to a sparse matrix for memory efficiency
    train_sparse = csr_matrix(train.values)

    # Perform k-means clustering with custom distance and memory efficiency
    centroids, clusters = kmeans_clustering_plus_plus(train_sparse.toarray(), num_clusters=k,
                                                      distance_func=cos_sim_mag_dist)

    # Save the centroids and clustering model for future use
    metadata_collection.replace_one(
        {'_id': 'kmeans_centroids'},
        {'_id': 'kmeans_centroids', 'centroids': centroids.tolist()},
        upsert=True
    )
    metadata_collection.replace_one(
        {'_id': 'kmeans_clusters'},
        {'_id': 'kmeans_clusters', 'clusters': clusters},
        upsert=True
    )

    print("Clustering model and centroids saved.")

    # Assign clusters to the training DB
    train.loc[:, 'cluster'] = [np.argmin([cos_sim_mag_dist(train.iloc[i].values, centroid) for centroid in centroids])
                               for i in range(len(train))]

    # Generate and save recommendations for each cluster
    def get_top_n_recommendations(df, top_n=40):
        """Get top N recommendations with their scores from a cluster DataFrame."""
        if df.empty:
            return [], []

        recommendations_with_scores = df.mean().sort_values(ascending=False).head(top_n)

        # Handle empty scores
        recommendations_with_scores = recommendations_with_scores.dropna()

        return recommendations_with_scores.index.tolist(), recommendations_with_scores.values.tolist()

    print("Generating and saving recommendations for each cluster...")

    cluster_recommendations = []
    n_clusters = k

    for cluster_id in range(n_clusters):
        cluster_df = train[train['cluster'] == cluster_id].drop('cluster', axis=1)
        recommendations, scores = get_top_n_recommendations(cluster_df)
        cluster_recommendations.append({
            '_id': ObjectId(),
            'cluster_id': cluster_id,
            'recommendations': [
                {'gmap_id': rec, 'score': score}
                for rec, score in zip(recommendations, scores)
            ]
        })

    # Save recommendations to MongoDB
    print("Saving recommendations to MongoDB...")
    clusters_collection.insert_many(cluster_recommendations)

    print("Recommendations saved to MongoDB.")

    # Calculate RMSE
    def predict_ratings(user_vector, centroids, cluster_recommendations):
        """Predict ratings based on user vector and cluster recommendations."""
        user_cluster = np.argmin([cos_sim_mag_dist(user_vector, centroid) for centroid in centroids])
        recommendations = next(
            (rec['recommendations'] for rec in cluster_recommendations if rec['cluster_id'] == user_cluster), [])
        rec_dict = {rec['gmap_id']: rec['score'] for rec in recommendations}
        return rec_dict

    def calculate_rmse(test_df, cluster_recommendations):
        """Calculate RMSE for predictions vs actual ratings."""
        total_squared_error = 0
        count = 0

        for user_id, user_ratings in test_df.iterrows():
            user_vector = user_ratings.values
            predictions = predict_ratings(user_vector, centroids, cluster_recommendations)
            actual_ratings = user_ratings[user_ratings > 0]

            for gmap_id, actual_rating in actual_ratings.items():
                if gmap_id in predictions:
                    predicted_rating = predictions[gmap_id]
                    squared_error = (actual_rating - predicted_rating) ** 2
                    total_squared_error += squared_error
                    count += 1

        if count == 0:
            return float('nan')
        return np.sqrt(total_squared_error / count)

    print("Calculating RMSE for the test set...")
    rmse = calculate_rmse(test, cluster_recommendations)
    print(f"Clustering Algorithm RMSE: {rmse}")


if __name__ == "__main__":
    run_clustering()
