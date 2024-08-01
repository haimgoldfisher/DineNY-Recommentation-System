#!/bin/bash
set -e
mkdir -p /dump/Google-Maps-Restaurant
mv /dump/Restaurants.bson /dump/Google-Maps-Restaurant/
mv /dump/Restaurants.metadata.json /dump/Google-Maps-Restaurant/
mv /dump/Reviews.bson /dump/Google-Maps-Restaurant/
mv /dump/Reviews.metadata.json /dump/Google-Maps-Restaurant/
mv /dump/Users.bson /dump/Google-Maps-Restaurant/
mv /dump/Users.metadata.json /dump/Google-Maps-Restaurant/
mongorestore /dump/Google-Maps-Restaurant
mongorestore --db Google-Maps-Restaurant --collection Restaurants /dump/Google-Maps-Restaurant/Restaurants.bson
mongorestore --db Google-Maps-Restaurant --collection Reviews /dump/Google-Maps-Restaurant/Reviews.bson
mongorestore --db Google-Maps-Restaurant --collection Users /dump/Google-Maps-Restaurant/Users.bson
echo "Restore completed successfully"
