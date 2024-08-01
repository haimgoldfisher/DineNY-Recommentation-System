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
