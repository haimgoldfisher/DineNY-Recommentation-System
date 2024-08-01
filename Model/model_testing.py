from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, collect_list, explode, desc
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexer
from pymongo import MongoClient

def update_user_recommendations(user_id):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Update User Recommendations") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Reviews") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Reviews") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.2") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .getOrCreate()

    print("Loading and preprocessing data...")

    # Load reviews and restaurants data from MongoDB
    df_reviews = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    df_restaurants = spark.read.format("com.mongodb.spark.sql.DefaultSource") \
        .option("uri", "mongodb://localhost:27017/Google-Maps-Restaurant.Restaurants") \
        .load()

    # Index the user_id and gmap_id columns in reviews data
    print("Indexing user and restaurant columns...")
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index").fit(df_reviews)
    rest_indexer = StringIndexer(inputCol="gmap_id", outputCol="rest_index").fit(df_reviews)

    df_reviews = user_indexer.transform(df_reviews)
    df_reviews = rest_indexer.transform(df_reviews)

    # Apply the same indexer to the restaurants data
    df_restaurants = rest_indexer.transform(df_restaurants)

    # Debug: Print schema to ensure fields are present
    df_reviews.printSchema()
    df_restaurants.printSchema()

    # Load the ALS model
    model = ALSModel.load("Model/Recommendation-Model")

    # Transform the user_id to user_index
    user_index = user_indexer.transform(spark.createDataFrame([(user_id,)], ["user_id"])).select("user_index").first()[
        "user_index"]

    # Generate recommendations for the specific user
    user_recommendations = model.recommendForUserSubset(spark.createDataFrame([(user_index,)], ["user_index"]), 13)

    # Explode recommendations into separate rows
    user_recommendations = user_recommendations.withColumn("recommendation", explode("recommendations"))

    # Debug: Print schema to ensure fields are present
    user_recommendations.printSchema()

    # Join with the original IDs
    user_recommendations = user_recommendations.join(df_reviews.select("user_index", "user_id").distinct(),
                                                     "user_index") \
        .join(df_restaurants.select("rest_index", "gmap_id").distinct(),
              user_recommendations["recommendation.rest_id"] == col("rest_index")) \
        .select("user_id", col("gmap_id"), col("recommendation.rating").alias("score"))

    # Group recommendations by user_id, sort by score, and collect them into an array
    user_recommendations_grouped = user_recommendations.orderBy(desc("score")) \
        .groupBy("user_id") \
        .agg(collect_list(struct("gmap_id", "score")).alias("recommendations"))

    # Write the DataFrame to a temporary MongoDB collection
    temp_collection = "TempRecommendations"
    user_recommendations_grouped.write \
        .format("com.mongodb.spark.sql.DefaultSource") \
        .mode("overwrite") \
        .option("collection", temp_collection) \
        .save()

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Google-Maps-Restaurant"]
    temp_coll = db[temp_collection]
    main_coll = db["Recommendations"]

    # Upsert recommendations into the main collection from the temporary collection
    for doc in temp_coll.find():
        main_coll.update_one(
            {"user_id": doc["user_id"]},
            {"$set": {"recommendations": doc["recommendations"]}},
            upsert=True
        )

    print(f"Recommendations for user {user_id} updated in MongoDB.")

    # Clean up temporary collection
    temp_coll.drop()

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    user_id = "100000036464530353686"  # Replace with the actual user_id
    update_user_recommendations(user_id)
