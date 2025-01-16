import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_results(folder_path, save_path):
    model_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    model_data = {}

    for model_dir in model_dirs:
        results_path = os.path.join(folder_path, model_dir, "results.csv")
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            model_data[model_dir] = {
                "epoch": df["epoch"],
                "mAP50": df["metrics/mAP50(B)"],
                "recall": df["metrics/recall(B)"]
            }

    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    for model_name, data in model_data.items():
        axs[0].plot(data["epoch"], data["mAP50"], label=model_name)
    axs[0].set_ylabel("mAP", fontsize=12)
    axs[0].set_title(r"Training mAP Results", fontsize=14)
    axs[0].legend(title="Models", fontsize=10, loc='best') 
    axs[0].grid(True)

    for model_name, data in model_data.items():
        axs[1].plot(data["epoch"], data["recall"], label=model_name)
    axs[1].set_xlabel("Epoch", fontsize=12)
    axs[1].set_ylabel("Recall", fontsize=12)
    axs[1].set_title(r"Training Recall Results", fontsize=14)
    axs[1].legend(title="Models", fontsize=10, loc='best')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

folder_path = "runs/detect/150 epoch/"
save_path = "figures/train_results.jpeg"
plot_train_results(folder_path, save_path)
