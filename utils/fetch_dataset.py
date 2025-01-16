import kagglehub

# Download latest version
path = kagglehub.dataset_download("koryakinp/d6-dices-images")

print("Path to dataset files:", path)