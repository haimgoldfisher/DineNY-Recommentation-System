from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Mongo URI getter from env. var
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("Restaurant Recommender") \
    .config("spark.mongodb.input.uri", f"{mongo_uri}Google-Maps-Restaurant.Reviews") \
    .config("spark.mongodb.output.uri", f"{mongo_uri}Google-Maps-Restaurant.Reviews") \
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

# Load data from MongoDB into Spark DataFrames
reviews_df = spark.read.format("mongo").load()
users_df = spark.read.format("mongo").option("uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Users").load()

# 1. Plot distribution of ratings
# Count the number of reviews per rating
rating_counts = reviews_df.groupBy("rating").count().orderBy("rating").toPandas()

# Plot distribution of ratings with Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x='rating', y='count', data=rating_counts, palette='Blues_d')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.xticks(rating_counts['rating'])
plt.savefig("plots/Rating_dist_seaborn.jpg")
plt.show()

# 2. Perform aggregations
# Total number of reviews
total_reviews = reviews_df.count()
print(f"Total number of reviews: {total_reviews}")

# Total number of users
total_users = users_df.count()
print(f"Total number of users: {total_users}")

# Average rating
average_rating = reviews_df.select(F.avg("rating")).first()[0]
print(f"Average rating across all reviews: {average_rating:.2f}")

# User with the most reviews
most_active_user = reviews_df.groupBy("user_id") \
    .agg(F.count("*").alias("count")) \
    .orderBy(F.desc("count")) \
    .first()
print(f"User with the most reviews: {most_active_user['user_id']} with {most_active_user['count']} reviews")

# Average number of reviews per user
avg_reviews_per_user = total_reviews / total_users
print(f"Average number of reviews per user: {avg_reviews_per_user:.2f}")

# 3. Distribution of Ratings for the 10 Most Reviewed Places
# Find the 10 most reviewed places
most_reviewed_places = reviews_df.groupBy("gmap_id") \
    .agg(F.count("*").alias("review_count")) \
    .orderBy(F.desc("review_count")) \
    .limit(10) \
    .join(reviews_df, on="gmap_id") \
    .groupBy("gmap_id", "rating") \
    .agg(F.count("*").alias("rating_count"))

# Convert to Pandas DataFrame for plotting
top_places_ratings = most_reviewed_places.toPandas()

# Plot distribution of ratings for the 10 most reviewed places with Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='rating', y='rating_count', hue='gmap_id', data=top_places_ratings, palette='Set2', dodge=True)
plt.title('Rating Distribution for Top 10 Most Reviewed Places')
plt.xlabel('Rating and Place ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.legend(title="Place ID")
plt.savefig("plots/Top_10_Most_Reviewed_Places_seaborn.jpg")
plt.show()

# 4. Top 10 Reviewers and Their Ratings
# Find the top 10 reviewers
top_reviewers = reviews_df.groupBy("user_id") \
    .agg(F.count("*").alias("review_count")) \
    .orderBy(F.desc("review_count")) \
    .limit(10) \
    .select("user_id")

# Join with the reviews DataFrame to get ratings for top reviewers
top_reviewers_reviews = top_reviewers.join(reviews_df, on="user_id") \
    .groupBy("user_id", "rating") \
    .agg(F.count("*").alias("rating_count"))

# Convert to Pandas DataFrame for plotting
top_reviewers_ratings = top_reviewers_reviews.toPandas()

# Plot distribution of ratings for the top 10 reviewers with Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='rating', y='rating_count', hue='user_id', data=top_reviewers_ratings, palette='Set1', dodge=True)
plt.title('Rating Distribution for Top 10 Reviewers')
plt.xlabel('Rating and Reviewer ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.legend(title="Reviewer ID")
plt.savefig("plots/Top_10_Reviewers_seaborn.jpg")
plt.show()