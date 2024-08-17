import json


def generate_gmap_id_query(file_path):
    # Load the gmap_ids from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract gmap_ids into a list
    gmap_id_list = [item['gmap_id'] for item in data]

    # Generate the MongoDB Compass query
    query = {"gmap_id": {"$in": gmap_id_list}}

    # Print the query in JSON format
    print(json.dumps(query, indent=4))


# Path to the JSON file
file_path = 'data/Google-Maps-Restaurant.Restaurants_under100.json'

# Generate and print the query
generate_gmap_id_query(file_path)
