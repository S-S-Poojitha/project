from bing_image_downloader import downloader

# Define the base query and colors
base_query = "frock"
colors = [
    "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown"
]

# Define the number of images to download per color
num_images = 10

# Define the output directory
output_dir = 'fashion_dataset'

# Download images for each color
for color in colors:
    query = f"{base_query} {color}"
    downloader.download(
        query,
        limit=num_images,
        output_dir=output_dir,
        adult_filter_off=False,  # Set to True if you want to include adult content
        force_replace=False,
        timeout=60,
        verbose=True
    )
