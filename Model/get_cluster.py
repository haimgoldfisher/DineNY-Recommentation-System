from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("Restaurant Recommender Test") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Users") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

print("Loading and preprocessing user data...")

# Load all users data from MongoDB
df_users = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

# Get 100 random users
random_users = df_users.select("user_id").distinct().sample(False, 0.01).limit(100).collect()
random_user_ids = [row["user_id"] for row in random_users]

# Load the trained K-Means model
kmeans_model = KMeansModel.load("K-Means-Model")

# Function to predict cluster for a single user
def predict_cluster_for_user(user_id):
    # Load new user data from MongoDB
    df_new_user = df_users.filter(col("user_id") == user_id)

    # Flatten the ratings array
    df_new_user = df_new_user.withColumn("rating", explode(col("ratings")))

    # Select relevant columns
    df_new_user_reviews = df_new_user.select(
        col("_id"),
        col("user_id"),
        col("rating.rating").alias("rating"),
        col("rating.gmap_id").alias("gmap_id")
    )

    # Extract user features
    new_user_features = df_new_user_reviews.groupBy("user_id").agg(avg("rating").alias("avg_rating"))
    assembler = VectorAssembler(inputCols=["avg_rating"], outputCol="features")
    new_user_features = assembler.transform(new_user_features)

    # Predict the cluster for the new user
    new_user_cluster = kmeans_model.transform(new_user_features).select("user_id", "prediction").collect()

    # Return the predicted cluster
    if new_user_cluster:
        return new_user_cluster[0]["prediction"]
    return None

# Iterate over 100 random users and predict their clusters
user_clusters = {}
for user_id in random_user_ids:
    cluster = predict_cluster_for_user(user_id)
    user_clusters[user_id] = cluster
    print(f"User {user_id} is assigned to cluster {cluster}")

# Stop the Spark session
spark.stop()