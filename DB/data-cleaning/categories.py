import json
from collections import defaultdict


def print_top_categories(file_path, top_n=50):
    category_counts = defaultdict(int)

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # skip empty lines
                try:
                    item = json.loads(line)
                    if 'category' in item and item['category'] is not None:
                        for category in item['category']:
                            category_counts[category] += 1
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except TypeError as e:
                    print(f"Error processing item: {e}")

    # Sort categories by count in descending order and get the top N
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    top_categories = sorted_categories[:top_n]

    for category, count in top_categories:
        print(f"{category}: {count}")


# Call the function with your file path
print_top_categories('data/meta-New_York.json')
