import json

# File paths
restaurants_file_path = 'data_2/Google-Maps-Restaurant.Restaurants.json'
reviews_file_path = 'data_2/Google-Maps-Restaurant.Reviews.json'
ids_g_file_path = 'data_2/ids_g.json'

# Load ids_g
print("Loading ids_g...")
with open(ids_g_file_path, 'r') as f:
    ids_g = json.load(f)
print(f"Loaded {len(ids_g)} ids to filter out.")

# Print first 10 ids_g for debugging
print("Sample gmap_id values from ids_g.json:")
for i in range(min(10, len(ids_g))):
    print(ids_g[i])

# Function to filter out documents by gmap_id
def filter_by_gmap_id(data, key):
    filtered_data = []
    count = 0
    for item in data:
        if item[key] not in ids_g:
            filtered_data.append(item)
        else:
            count += 1
            if count <= 10:  # Print the first 10 filtered gmap_ids for debugging
                print(f"Filtering out item with gmap_id: {item[key]}")
    print(f"Total items filtered out: {count}")
    return filtered_data

# Load and filter Restaurants
print("Loading Restaurants data...")
with open(restaurants_file_path, 'r') as f:
    restaurants = json.load(f)
print(f"Loaded {len(restaurants)} restaurants.")

# Print first 10 gmap_id values from restaurants for debugging
print("Sample gmap_id values from Restaurants data:")
for i in range(min(10, len(restaurants))):
    print(restaurants[i]['gmap_id'])

print("Filtering Restaurants data...")
filtered_restaurants = filter_by_gmap_id(restaurants, 'gmap_id')
print(f"Filtered Restaurants data. {len(filtered_restaurants)} remaining.")

# Load and filter Reviews
print("Loading Reviews data...")
with open(reviews_file_path, 'r') as f:
    reviews = json.load(f)
print(f"Loaded {len(reviews)} reviews.")

print("Filtering Reviews data...")
filtered_reviews = filter_by_gmap_id(reviews, 'gmap_id')
print(f"Filtered Reviews data. {len(filtered_reviews)} remaining.")

# Save the filtered data back to JSON files
print("Saving filtered data...")

with open(restaurants_file_path, 'w') as f:
    json.dump(filtered_restaurants, f, indent=4)
print(f"Updated Restaurants data saved to {restaurants_file_path}.")

with open(reviews_file_path, 'w') as f:
    json.dump(filtered_reviews, f, indent=4)
print(f"Updated Reviews data saved to {reviews_file_path}.")

print("Filtering complete and files have been updated.")
