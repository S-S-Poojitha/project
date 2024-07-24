from bing_image_downloader import downloader

# Define the base queries for different objects
base_queries = [
    "decorations",
    "phones",
    "bags",
    "watches",
    "shoes"
]

# Define the colors to search for
colors = [
    "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown"
]

# Define the number of images to download per query and color
num_images = 10

# Define the output directory
output_dir = 'fashion_dataset'

# Download images for each query and color
for base_query in base_queries:
    for color in colors:
        query = f"{base_query} {color}"
        downloader.download(
            query,
            limit=num_images,
            output_dir=output_dir,
            adult_filter_off=True,  # Exclude adult content
            force_replace=False,
            timeout=60,
            verbose=True
        )
