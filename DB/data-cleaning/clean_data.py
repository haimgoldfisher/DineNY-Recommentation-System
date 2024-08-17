import json

# Path to the input JSON file and the restaurant IDs file
input_file_path = 'data/review-New_York_10.json'
restaurants_file_path = 'data/Restaurants_ids'
output_file_path = 'data/reviews.json'

# Read the restaurant IDs from the text file into a set for fast lookup
with open(restaurants_file_path, 'r') as text_file:
    gmap_ids_in_file = set(line.strip() for line in text_file)

# Prepare a list to store cleaned data
cleaned_data_list = []

# Read and process each JSON object in the input file
with open(input_file_path, 'r') as infile:
    for line in infile:
        if line.strip():  # Skip empty lines
            try:
                data_dict = json.loads(line)
                # Extract the required fields
                gmap_id = data_dict.get("gmap_id")
                # Check if the gmap_id from the review is in the set
                if gmap_id in gmap_ids_in_file:
                    cleaned_data = {
                        "user_id": data_dict.get("user_id"),
                        "time": data_dict.get("time"),
                        "rating": data_dict.get("rating"),
                        "gmap_id": gmap_id
                    }
                    cleaned_data_list.append(cleaned_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}\n{e}")

# Save the cleaned data to a new JSON file
with open(output_file_path, 'w') as outfile:
    json.dump(cleaned_data_list, outfile, indent=4)

print(f"Cleaned data saved to {output_file_path}")
