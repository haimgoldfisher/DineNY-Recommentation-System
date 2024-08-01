Pull the image (if it's not already on your local machine):

```shell
docker pull haimgoldfisher/google-maps-restaurants-reviews-db:1.2
```

Run the container using the pulled image:
```shell
docker run -d -p 27017:27017 --name google-maps-mongo haimgoldfisher/google-maps-restaurants-reviews-db:1.0
```
