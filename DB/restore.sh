#!/bin/bash
set -e
mkdir -p /dump/Google-Maps-Restaurant
mv /dump/Restaurants.bson /dump/Google-Maps-Restaurant/
mv /dump/Restaurants.metadata.json /dump/Google-Maps-Restaurant/
mv /dump/Reviews.bson /dump/Google-Maps-Restaurant/
mv /dump/Reviews.metadata.json /dump/Google-Maps-Restaurant/
mv /dump/Users.bson /dump/Google-Maps-Restaurant/
mv /dump/Users.metadata.json /dump/Google-Maps-Restaurant/
mv /dump/Recommendations.bson /dump/Google-Maps-Restaurant/
mv /dump/Recommendations.metadata.json /dump/Google-Maps-Restaurant/
mv /dump/Clusters.bson /dump/Google-Maps-Restaurant/
mv /dump/Clusters.metadata.json /dump/Google-Maps-Restaurant/
mv /dump/Metadata.bson /dump/Google-Maps-Restaurant/
mv /dump/Metadata.metadata.json /dump/Google-Maps-Restaurant/
mongorestore /dump/Google-Maps-Restaurant
mongorestore --db Google-Maps-Restaurant --collection Restaurants /dump/Google-Maps-Restaurant/Restaurants.bson
mongorestore --db Google-Maps-Restaurant --collection Reviews /dump/Google-Maps-Restaurant/Reviews.bson
mongorestore --db Google-Maps-Restaurant --collection Users /dump/Google-Maps-Restaurant/Users.bson
mongorestore --db Google-Maps-Restaurant --collection Recommendations /dump/Google-Maps-Restaurant/Recommendations.bson
mongorestore --db Google-Maps-Restaurant --collection Clusters /dump/Google-Maps-Restaurant/Clusters.bson
mongorestore --db Google-Maps-Restaurant --collection Metadata /dump/Google-Maps-Restaurant/Metadata.bson
echo "Restore completed successfully"