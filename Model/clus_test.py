import numpy as np
from pymongo import MongoClient
from dists import cos_sim_mag_dist as dist


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

# MongoDB connection details
client = MongoClient('mongodb://localhost:27017/')
db = client['Google-Maps-Restaurant']
users_collection = db['Users']
clusters_collection = db['Clusters']

def format_user_data(user_data, columns):
    """Format user data into the same format used during training."""
    user_ratings = {item['gmap_id']: item['rating'] for item in user_data['ratings'] if 'rating' in item}
    formatted_data = np.array([user_ratings.get(col, 0) for col in columns])
    return formatted_data

def get_user_cluster(user_vector, centroids):
    """Predict the cluster for a given user vector by calculating distances to centroids."""
    distances = [dist(user_vector, centroid) for centroid in centroids]
    cluster = np.argmin(distances)
    print(f"user got cluster # {cluster}")
    return cluster

def get_user_recommendations(user_cluster):
    """Fetch recommendations for the user's cluster from MongoDB."""
    # Convert numpy types to native Python types
    user_cluster = int(user_cluster)

    cluster_doc = clusters_collection.find_one({'cluster_id': user_cluster})
    if cluster_doc:
        return cluster_doc['recommendations']
    return []

def main(user_id):
    """Main function to handle recommendation for a user."""
    # Load KMeans centroids and feature columns from MongoDB
    columns, centroids = load_from_mongodb()

    # Load user data from MongoDB
    user_data = users_collection.find_one({'user_id': user_id})
    if not user_data:
        raise ValueError(f"No data found for user with ID {user_id}")

    # Format the user data
    user_vector = format_user_data(user_data, columns)

    # Get the user's cluster
    user_cluster = get_user_cluster(user_vector, centroids)

    # Get recommendations for the user's cluster
    recommendations = get_user_recommendations(user_cluster)

    return recommendations

if __name__ == "__main__":
    # Example user ID to test
    user_id = "100000125622723157419"
    recommendations = main(user_id)
    print(f"Recommendations for user {user_id}: {recommendations}")
