FROM mongo:latest
LABEL authors="haimgoldfisher" version=1.2

# Copy dump files
COPY ./dump /dump

# Change permissions and ownership of the dump directory and its contents
RUN chown -R mongodb:mongodb /dump && chmod -R 755 /dump

# Copy and set permissions for the restore script
COPY restore.sh /docker-entrypoint-initdb.d/restore.sh
RUN chmod +x /docker-entrypoint-initdb.d/restore.sh

USER mongodb

CMD ["mongod", "--bind_ip_all"]

# Option B: didnt triec yet:
# Use the official MongoDB image
#FROM mongo:latest
#
#LABEL authors="haimgoldfisher" version=1.2
#
## Copy dump files into the container
#COPY ./dump /dump
#
## Change permissions and ownership of the dump directory and its contents
#RUN chown -R mongodb:mongodb /dump && chmod -R 755 /dump
#
## Copy the restore script to the init directory
#COPY restore.sh /docker-entrypoint-initdb.d/restore.sh
#
## Make the restore script executable
#RUN chmod +x /docker-entrypoint-initdb.d/restore.sh
#
## Use the MongoDB user
#USER mongodb
#
## Start MongoDB
#CMD ["mongod", "--bind_ip_all"]

