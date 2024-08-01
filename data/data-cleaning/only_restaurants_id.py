import json

def get_gmap_ids_with_category(file_path, target_category):
    gmap_ids = set()

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # skip empty lines
                try:
                    item = json.loads(line)
                    if 'category' in item and item['category'] is not None:
                        if target_category in item['category']:
                            if 'gmap_id' in item:
                                gmap_ids.add(item['gmap_id'])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except TypeError as e:
                    print(f"Error processing item: {e}")

    # Join the gmap_ids with a newline character
    gmap_ids_text = '\n'.join(gmap_ids)
    print(gmap_ids)
    print(len(gmap_ids))
    return gmap_ids_text

# Call the function with your file path and target category
result_text = get_gmap_ids_with_category('data/meta-New_York.json', 'Restaurant')
with open("data/Restaurants_ids", 'w') as output_file:
    output_file.write(result_text)

