import json

# Path to the input JSON file and the output file
input_file_path = 'data/meta-New_York.json'
restaurants_file_path = 'data/Restaurants_ids'
output_file_path = 'data/metadata.json'

# Read the text file content into a set for fast lookup
with open(restaurants_file_path, 'r') as text_file:
    gmap_ids_in_file = set(line.strip() for line in text_file)

# Prepare a list to hold filtered metadata
filtered_metadata_list = []

# Process each line of the JSON file
with open(input_file_path, 'r') as json_file:
    for line in json_file:
        try:
            # Load the JSON object from the current line
            metadata = json.loads(line)

            # Check if the gmap_id from metadata is in the set
            if metadata["gmap_id"] in gmap_ids_in_file:
                # Filtered metadata
                filtered_metadata = {
                    "name": metadata["name"],
                    "address": metadata["address"],
                    "gmap_id": metadata["gmap_id"],
                    "url": metadata["url"]
                }
                filtered_metadata_list.append(filtered_metadata)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

# Write filtered metadata to the output file
with open(output_file_path, 'w') as output_file:
    json.dump(filtered_metadata_list, output_file, indent=4)

print(f"Filtered metadata has been written to {output_file_path}")
