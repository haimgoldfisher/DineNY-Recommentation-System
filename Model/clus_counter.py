import numpy as np
from pymongo import MongoClient
from dists import cos_sim_mag_dist as dist
from collections import Counter

def load_from_mongodb():
    """Load KMeans centroids and feature columns from MongoDB."""
    # MongoDB connection details
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Google-Maps-Restaurant']
    metadata_collection = db['Metadata']

    # Load the feature columns
    columns_doc = metadata_collection.find_one({'_id': 'train_columns'})
    columns = columns_doc['columns'] if columns_doc else []

    # Load the centroids
    centroids_doc = metadata_collection.find_one({'_id': 'kmeans_centroids'})
    centroids = np.array(centroids_doc['centroids']) if centroids_doc else np.array([])

    return columns, centroids

def format_user_data(user_data, columns):
    """Format user data into the same format used during training."""
    user_ratings = {item['gmap_id']: item['rating'] for item in user_data['ratings'] if 'rating' in item}
    # Reindex user data to match the columns used in training
    formatted_data = np.array([user_ratings.get(col, 0) for col in columns])
    return formatted_data

def get_user_cluster(user_vector, centroids):
    """Predict the cluster for a given user vector by calculating distances to centroids."""
    distances = [dist(user_vector, centroid) for centroid in centroids]
    return np.argmin(distances)

def count_users_per_cluster():
    """Count the number of users in each cluster."""
    cluster_counts = Counter()

    # MongoDB connection details
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Google-Maps-Restaurant']
    users_collection = db['Users']

    # Load KMeans centroids and feature columns from MongoDB
    columns, centroids = load_from_mongodb()

    # Fetch all users from the database
    all_users = users_collection.find()
    i = 0
    for user_data in all_users:
        if i > 1000:
            continue
        # Format the user data
        user_vector = format_user_data(user_data, columns)

        # Get the user's cluster
        user_cluster = get_user_cluster(user_vector, centroids)

        # Increment the counter for this cluster
        cluster_counts[user_cluster] += 1
        i = i + 1

    return cluster_counts

if __name__ == "__main__":
    # Get the counts of users in each cluster
    cluster_counts = count_users_per_cluster()
    # Print out the number of users per cluster
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} users")
