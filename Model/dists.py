import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def compute_similarity_for_pair(i, j, matrix_sparse, magnitudes):
    user_i = matrix_sparse.getrow(i).toarray().flatten()
    user_j = matrix_sparse.getrow(j).toarray().flatten()

    # Create mask for non-zero interactions
    mask = (user_i > 0) & (user_j > 0)

    if np.any(mask):
        # Apply the mask
        user_i_masked = user_i[mask]
        user_j_masked = user_j[mask]

        # Compute cosine similarity
        dot_product = np.dot(user_i_masked, user_j_masked)
        magnitude_i = magnitudes[i]
        magnitude_j = magnitudes[j]
        cosine_similarity = dot_product / (magnitude_i * magnitude_j)

        # Compute magnitude differences
        magnitude_diff = abs(magnitude_i - magnitude_j)
        max_magnitude = max(magnitude_i, magnitude_j)

        # Compute adjusted similarity
        adjusted_similarity = cosine_similarity * (1 - (magnitude_diff / max_magnitude))
        return (i, j, adjusted_similarity)
    else:
        return (i, j, 0)


def compute_similarity_parallel(matrix):
    # Convert matrix to sparse format
    matrix_sparse = csr_matrix(matrix)

    # Compute magnitudes
    magnitudes = np.sqrt(matrix_sparse.multiply(matrix_sparse).sum(axis=1)).A1

    # Prepare similarity matrix as a sparse matrix (LIL format for efficient assignment)
    num_users = matrix_sparse.shape[0]
    similarity_matrix = lil_matrix((num_users, num_users))

    # Compute similarities in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_similarity_for_pair)(i, j, matrix_sparse, magnitudes)
                                  for i in range(num_users) for j in range(i, num_users))

    # Fill the similarity matrix with results
    for (i, j, similarity) in results:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # Symmetric matrix

    # Convert to CSR format for efficient computation
    return similarity_matrix.tocsr()


def cos_sim_mag_dist(user_i, user_j):
    # Create a mask for non-zero interactions in both users
    non_zero_mask = (user_i != 0) & (user_j != 0)

    # Apply the mask to both user vectors
    masked_user_i = user_i[non_zero_mask]
    masked_user_j = user_j[non_zero_mask]

    # If there are no common non-zero items, return max distance (1)
    if len(masked_user_i) == 0 or len(masked_user_j) == 0:
        # print("No common non-zero items between users.")
        return 1

    # Compute cosine similarity on masked vectors
    dot_product = np.dot(masked_user_i, masked_user_j)
    magnitude_i = np.sqrt(np.sum(masked_user_i ** 2))
    magnitude_j = np.sqrt(np.sum(masked_user_j ** 2))

    # Handle zero magnitude vectors after masking
    if magnitude_i == 0 or magnitude_j == 0:
        print("One of the vectors has zero magnitude.")
        return 1  # Max distance if one of the vectors has zero magnitude

    cosine_similarity = dot_product / (magnitude_i * magnitude_j)

    # Compute magnitude difference
    magnitude_diff = abs(magnitude_i - magnitude_j)

    # Compute adjusted similarity
    adjusted_similarity = cosine_similarity * (1 - (magnitude_diff / max(magnitude_i, magnitude_j)))

    # Convert similarity to distance (1 - similarity)
    adjusted_distance = 1 - adjusted_similarity
    return adjusted_distance



def assign_clusters(users_matrix, centroids, distance_func):
    clusters = [[] for _ in range(len(centroids))]
    total_error = 0
    for i, user_vector in enumerate(users_matrix):
        distances = [distance_func(user_vector, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(i)
        total_error += distances[closest_centroid]  # Track total error
    return clusters, total_error


def update_centroids(users_matrix, clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Check if cluster is non-empty
            new_centroids.append(np.mean(users_matrix[cluster], axis=0))
        else:
            new_centroids.append(np.zeros(users_matrix.shape[1]))  # Handle empty clusters
    return np.array(new_centroids)


def kmeans_clustering(users_matrix, num_clusters, distance_func, max_iters=100, tol=1e-4):
    # Initialize centroids randomly
    centroids = users_matrix[np.random.choice(range(len(users_matrix)), num_clusters, replace=False)]

    prev_error = None

    for _ in range(max_iters):
        clusters, total_error = assign_clusters(users_matrix, centroids, distance_func)
        new_centroids = update_centroids(users_matrix, clusters)

        # Check convergence based on total error
        if prev_error is not None and abs(prev_error - total_error) < tol:
            break  # Stop if the change in error is less than the tolerance
        prev_error = total_error

        if np.allclose(centroids, new_centroids):
            break  # Stop if centroids have converged

        centroids = new_centroids

    return centroids, clusters


def kmeans_clustering_plus_plus(users_matrix, num_clusters, distance_func, max_iters=100, tol=1e-4):
    # Initialize centroids using k-means++
    kmeans_init = KMeans(n_clusters=num_clusters, init='k-means++', n_init=1).fit(users_matrix)
    centroids = kmeans_init.cluster_centers_

    prev_error = None

    for _ in range(max_iters):
        clusters, total_error = assign_clusters(users_matrix, centroids, distance_func)
        new_centroids = update_centroids(users_matrix, clusters)

        # Check convergence based on total error
        if prev_error is not None and abs(prev_error - total_error) < tol:
            break  # Stop if the change in error is less than the tolerance
        prev_error = total_error

        if np.allclose(centroids, new_centroids):
            break  # Stop if centroids have converged

        centroids = new_centroids

    return centroids, clusters


def calculate_silhouette_score(users_matrix, clusters, distance_func):
    labels = np.zeros(users_matrix.shape[0], dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for user_idx in cluster:
            labels[user_idx] = cluster_idx

    # Calculate silhouette score
    score = silhouette_score(users_matrix, labels, metric=distance_func)
    return score
