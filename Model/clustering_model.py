from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, avg, struct, collect_list
from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("Restaurant Recommender") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Reviews") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Clusters") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "200s") \
    .getOrCreate()

print("Loading and preprocessing data...")

# Load data from MongoDB
df_reviews = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

# Index the user_id and gmap_id columns
print("Indexing user and restaurant columns...")
user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index").fit(df_reviews)
rest_indexer = StringIndexer(inputCol="gmap_id", outputCol="rest_id").fit(df_reviews)
df_reviews = user_indexer.transform(df_reviews)
df_reviews = rest_indexer.transform(df_reviews)

# Extract user features for clustering
print("Extracting user features...")
user_features = df_reviews.groupBy("user_index").agg(avg("rating").alias("avg_rating"))
assembler = VectorAssembler(inputCols=["avg_rating"], outputCol="features")
user_features = assembler.transform(user_features)
user_features.show()

# Apply K-Means clustering
print("Clustering users...")
k = 5  # Number of clusters
kmeans = KMeans(k=k, seed=555, featuresCol="features", predictionCol="prediction")
kmeans_model = kmeans.fit(user_features)
user_features = kmeans_model.transform(user_features)

# Rename the column from 'prediction' to 'prediction' for ClusteringEvaluator
user_features = user_features.withColumnRenamed("prediction", "prediction")

# Evaluate clustering
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(user_features)
print(f"Silhouette with squared euclidean distance = {silhouette}")

# Save the clustering model
kmeans_model.write().overwrite().save("K-Means-Model")

# Join cluster info with original review data
df_with_clusters = df_reviews.join(user_features.select("user_index", "prediction"), on="user_index")

# Generate recommendations for each cluster
print("Generating recommendations for each cluster...")
cluster_recommendations = df_with_clusters.groupBy("prediction", "rest_id").agg(avg("rating").alias("avg_rating"))

# Join with original data to get gmap_id
cluster_recommendations = cluster_recommendations.join(df_reviews.select("rest_id", "gmap_id").distinct(), on="rest_id")

# Replace the rest_id with gmap_id in the recommendation struct
cluster_recommendations = cluster_recommendations.withColumn("recommendation", struct(col("gmap_id").alias("gmap_id"), col("avg_rating")))

# Get top 40 recommendations for each cluster
top_recommendations = cluster_recommendations.groupBy("prediction") \
    .agg(collect_list("recommendation").alias("recommendations")) \
    .select("prediction", col("recommendations"))

# Sort recommendations for each cluster by rating and take top 40
top_recommendations = top_recommendations.rdd.map(lambda row: (row[0], sorted(row[1], key=lambda x: -x.avg_rating)[:40])).toDF(["cluster_id", "recommendations"])

# Write the recommendations to MongoDB
print("Saving recommendations to MongoDB...")
top_recommendations.write \
    .format("com.mongodb.spark.sql.DefaultSource") \
    .mode("overwrite") \
    .option("collection", "Clusters") \
    .save()

print("Recommendations saved to MongoDB.")

# Stop the Spark session
spark.stop()