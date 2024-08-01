from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, explode, collect_list, struct

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("Restaurant Recommender") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Reviews") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Recommendations") \
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

# Check the initial data
df_reviews.show()

# Index the user_id and gmap_id columns
print("Indexing user and restaurant columns...")

user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index").fit(df_reviews)
rest_indexer = StringIndexer(inputCol="gmap_id", outputCol="rest_id").fit(df_reviews)

df_reviews = user_indexer.transform(df_reviews)
df_reviews = rest_indexer.transform(df_reviews)

# Select relevant columns and split into training and test sets
df_final = df_reviews.select("user_index", "rest_id", "rating").filter(df_reviews.rating.isNotNull())
df_final = df_final.repartition(200)

train, test = df_final.randomSplit([0.8, 0.2])

print("Transforming training data...")
train.show()

# Train the ALS model
print("Training the ALS model...")
als = ALS(maxIter=10, regParam=0.1, userCol="user_index", itemCol="rest_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train)

print("Training complete. Testing the model on the test set...")

# Test the model
predictions = model.transform(test)
predictions.show()

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Save the model with overwrite option
model.write().overwrite().save("Model/Recommendation-Model")

# Generate top 13 recommendations for each user
user_recs = model.recommendForAllUsers(13)

# Explode recommendations into separate rows
user_recs = user_recs.withColumn("recommendation", explode("recommendations"))

# Join with the original IDs
user_recs = user_recs.join(df_reviews.select("user_index", "user_id").distinct(), "user_index") \
    .join(df_reviews.select("rest_id", "gmap_id").distinct(), user_recs["recommendation.rest_id"] == col("rest_id")) \
    .select("user_id", col("recommendation.rest_id").alias("rest_id"), col("recommendation.rating").alias("score")) \
    .join(df_reviews.select("rest_id", "gmap_id").distinct(), "rest_id") \
    .select("user_id", col("gmap_id"), col("score"))

# Group recommendations by user_id and collect them into an array
user_recs_grouped = user_recs.groupBy("user_id") \
    .agg(collect_list(struct("gmap_id", "score")).alias("recommendations"))

# Write the DataFrame directly to MongoDB
user_recs_grouped.write \
    .format("com.mongodb.spark.sql.DefaultSource") \
    .mode("overwrite") \
    .option("collection", "Recommendations") \
    .save()

print("Recommendations saved to MongoDB.")

# Stop the Spark session
spark.stop()
