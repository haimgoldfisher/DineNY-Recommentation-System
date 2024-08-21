# MongoDB Service

## Overview

This project sets up a MongoDB service using the official MongoDB Docker image. The service is configured to initialize with a predefined dataset, ensuring that the database starts with the necessary data.

## Dockerfile Details

The Dockerfile used to build the MongoDB Docker image includes the following steps:

1. **Base Image**: Uses the latest MongoDB image from Docker Hub.
2. **Data Preparation**:
   - Copies MongoDB dump files into the container. These files are essential for initializing the database with predefined collections and data.
   - Changes the ownership and permissions of the dump directory to ensure MongoDB can access and use the files.
3. **Restore Script**:
   - Includes a script named `restore.sh` that is automatically executed when the container starts. This script is responsible for restoring the MongoDB database from the provided dump files.
4. **MongoDB Initialization**:
   - Configures MongoDB to bind to all network interfaces, making the service accessible from outside the container.

### Included Files

- **Dump Files**: Contains BSON and metadata files for various collections, including:
  - `Clusters.bson`, `Metadata.bson`, `Restaurants.bson`, `Recommendations.bson`, `Reviews.bson`, `Users.bson`
  - Metadata files for each collection to describe the structure and content of the data.
- **Restore Script**: A script placed in `/docker-entrypoint-initdb.d/` that handles database restoration during container initialization.

## Service Configuration

The MongoDB service is designed to start with the preloaded data, providing an out-of-the-box database setup with the necessary collections and initial data. This setup ensures that the MongoDB instance is ready for use immediately after startup.

For deployment and usage, the service listens on port 27017, which is the default port for MongoDB.

## Initialization and Access

The MongoDB instance will automatically initialize with the provided data once the container is started. You can interact with the MongoDB service using standard MongoDB tools and drivers, connecting to it through the container's exposed port.

## Notes

- Ensure that any necessary configurations or scripts are correctly placed in the Docker image to match your specific requirements.
- The `restore.sh` script should handle the restoration of the database from the dump files, so verify that this script is correctly set up and tested.
- For more information about the dataset, see the main README.md file under Dataset section.

For any further customization or issues, please refer to the MongoDB documentation or consult the project repository's resources.
