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
mongorestore --db Google-Maps-Restaurant --collection Recommendations /dump/Google-Maps-Restaurant/Recommendations.bson
mongorestore --db Google-Maps-Restaurant --collection Clusters /dump/Google-Maps-Restaurant/Clusters.bson
mongorestore --db Google-Maps-Restaurant --collection Metadata /dump/Google-Maps-Restaurant/Metadata.bson
echo "Restore completed successfully"


# OPTION B:

##!/bin/bash
#set -e
#
## Create directory for the database dump
#mkdir -p /dump/Google-Maps-Restaurant
#
## Move BSON and metadata files into the appropriate directory
#mv /dump/Restaurants.bson /dump/Google-Maps-Restaurant/
#mv /dump/Restaurants.metadata.json /dump/Google-Maps-Restaurant/
#mv /dump/Reviews.bson /dump/Google-Maps-Restaurant/
#mv /dump/Reviews.metadata.json /dump/Google-Maps-Restaurant/
#mv /dump/Users.bson /dump/Google-Maps-Restaurant/
#mv /dump/Users.metadata.json /dump/Google-Maps-Restaurant/
#
## If additional collections exist, move them as well
#if [ -f /dump/Recommendations.bson ]; then
#  mv /dump/Recommendations.bson /dump/Google-Maps-Restaurant/
#fi
#
#if [ -f /dump/Clusters.bson ]; then
#  mv /dump/Clusters.bson /dump/Google-Maps-Restaurant/
#fi
#
#if [ -f /dump/Metadata.bson ]; then
#  mv /dump/Metadata.bson /dump/Google-Maps-Restaurant/
#fi
#
## Restore the database dump
#mongorestore /dump/Google-Maps-Restaurant
#
## Restore completed
#echo "Restore completed successfully"

