import os
import matplotlib.pyplot as plt
from collections import defaultdict

dataset_path = "..\dataset\dataset3"
labels_folder = os.path.join(dataset_path, "labels")

train_file = os.path.join(dataset_path, "train.txt")
valid_file = os.path.join(dataset_path, "valid.txt")

figures_path = r"..\figures"
os.makedirs(figures_path, exist_ok=True)

def read_split_distribution(split_file, labels_folder):
    class_counts = {i: 0 for i in range(1, 7)}
    image_count = 0

    try:
        with open(split_file, "r") as file:
            image_paths = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"The split file '{split_file}' does not exist.")

    for image_path in image_paths:
        image_path = image_path.strip()
        label_path = os.path.join(labels_folder, os.path.basename(image_path).replace(".jpg", ".txt"))

        if os.path.exists(label_path):
            image_count += 1
            try:
                with open(label_path, "r") as label_file:
                    for line in label_file:
                        parts = line.split()
                        if len(parts) > 0:
                            class_id = int(parts[0]) + 1  # Shift from 0-5 to 1-6
                            if 1 <= class_id <= 6:
                                class_counts[class_id] += 1
                            else:
                                print(f"Warning: Class ID {class_id} out of range in file {label_path}")
            except Exception as e:
                print(f"Error reading label file '{label_path}': {e}")
        else:
            print(f"Warning: Label file '{label_path}' does not exist.")
    
    return class_counts, image_count

train_class_counts, train_image_count = read_split_distribution(train_file, labels_folder)
valid_class_counts, valid_image_count = read_split_distribution(valid_file, labels_folder)

def combine_class_distributions(train_counts, valid_counts):
    combined_counts = defaultdict(int)
    for class_id in train_counts:
        combined_counts[class_id] = train_counts[class_id] + valid_counts.get(class_id, 0)
    return combined_counts

combined_class_counts = combine_class_distributions(train_class_counts, valid_class_counts)
combined_image_count = train_image_count + valid_image_count

def plot_combined_class_distributions(counts, total_images, save_path):
    """
    Plots combined class distributions for train and validation splits, highlighting Class 5 in red.

    Args:
        counts (dict): A dictionary of combined class counts.
        total_images (int): Total number of images in the combined splits.
        save_path (str): Path to save the combined plot.
    """
    class_ids = list(counts.keys())
    frequencies = list(counts.values())

    colors = ['red' if class_id == 5 else 'green' for class_id in class_ids] # Highlithgs Class 5 in red to demonstrate class imbalance

    plt.figure(figsize=(8, 6))
    plt.barh(class_ids, frequencies, color=colors, edgecolor='black')
    plt.xlabel("Frequency")
    plt.ylabel("Class ID")
    plt.title(f"Class Balance\nTotal Images: {total_images}, Total Labels: {sum(frequencies)}")
    plt.yticks(class_ids)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

save_path = os.path.join(figures_path, "combined_distribution.png")
plot_combined_class_distributions(combined_class_counts, combined_image_count, save_path)

print("Combined Class Distributions:")
print(combined_class_counts)
print(f"Total Images: {combined_image_count}")
