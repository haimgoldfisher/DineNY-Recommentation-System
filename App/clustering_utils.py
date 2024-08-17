import numpy as np


def cos_sim_mag_dist(user_i, user_j):
    # Create a mask for non-zero interactions in both users
    non_zero_mask = (user_i != 0) & (user_j != 0)
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
