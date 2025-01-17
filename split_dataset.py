import os
import random
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_FOLDER = "dataset/dice-dataset/"
IMAGES_FOLDER = os.path.join(DATA_FOLDER, "images")
LABELS_FOLDER = os.path.join(DATA_FOLDER, "labels")

OUTPUT_FOLDER = "split_dataset"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TRAIN_FILE = os.path.join(OUTPUT_FOLDER, "train.txt")
VALID_FILE = os.path.join(OUTPUT_FOLDER, "valid.txt")

FIGURES_FOLDER = "figures"
os.makedirs(FIGURES_FOLDER, exist_ok=True)

SPLIT_RATIOS = {"train": 0.9, "valid": 0.1} # Split Ratio

images = sorted([os.path.join(IMAGES_FOLDER, img) for img in os.listdir(IMAGES_FOLDER) if img.endswith(".jpg")])
labels = sorted([os.path.join(LABELS_FOLDER, lbl) for lbl in os.listdir(LABELS_FOLDER) if lbl.endswith(".txt")])

data_pairs = list(zip(images, labels))

data_by_class = defaultdict(list)
for image_path, label_path in data_pairs:
    with open(label_path, "r") as f:
        for line in f:
            class_id = int(line.split()[0]) + 1  # Shifts Det IDs: 0-5 to 1-6
            if 1 <= class_id <= 6:
                data_by_class[class_id].append((image_path, label_path))
                break


train_data, valid_data = [], []
for class_id, items in data_by_class.items():
    random.shuffle(items)
    total = len(items)
    train_end = int(total * SPLIT_RATIOS["train"])

    train_data.extend(items[:train_end])
    valid_data.extend(items[train_end:])

def write_split(file_path, split_data):
    with open(file_path, "w") as f:
        for image, _ in split_data:
            f.write(f"{image}\n")


write_split(TRAIN_FILE, train_data)
write_split(VALID_FILE, valid_data)

def count_classes(split_data):
    class_counts = {i: 0 for i in range(1, 7)}
    image_count = 0

    for _, label_path in split_data:
        image_count += 1
        with open(label_path, "r") as file:
            for line in file:
                class_id = int(line.split()[0]) + 1  # Shift from 0-5 to 1-6
                if 1 <= class_id <= 6:
                    class_counts[class_id] += 1

    return class_counts, image_count

train_class_counts, train_image_count = count_classes(train_data)
valid_class_counts, valid_image_count = count_classes(valid_data)

def combine_class_distributions(train_counts, valid_counts):
    combined_counts = defaultdict(int)
    for class_id in train_counts:
        combined_counts[class_id] = train_counts[class_id] + valid_counts.get(class_id, 0)
    return combined_counts

combined_class_counts = combine_class_distributions(train_class_counts, valid_class_counts)
combined_image_count = train_image_count + valid_image_count

def plot_combined_class_distributions(counts, total_images, save_path):
    class_ids = list(counts.keys())
    frequencies = list(counts.values())

    colors = ['red' if class_id == 5 else 'green' for class_id in class_ids] # Dice-dataset: Visualizing Class Imbalance for Dice 5

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

save_path = os.path.join(FIGURES_FOLDER, "combined_distribution.png")
plot_combined_class_distributions(combined_class_counts, combined_image_count, save_path)

print("Class Distributions for Train Split:")
print(train_class_counts)

print("\nClass Distributions for Validation Split:")
print(valid_class_counts)

print("\nCombined Class Distributions:")
print(combined_class_counts)
print(f"Total Images: {combined_image_count}")
