from ultralytics import YOLO
import numpy as np

def load_model(model_path=None):
    """
        Description:
        Returns the loaded Computer Vision model using Ultralytics
        
        Returns:
            model: YOLO
            status: int
    """
    if not model_path:
        print("Warning: Model path not set")
        return None, -2
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print("Error when loading the model", e)
        return None, -1

def load_sam(model_path=None): # Not Used...
    """ Loads SAM model... 
        Not used
    """
    return SAM(model_path)


def detect(model, frame, device='cpu'):
    """
        Inference using CPU or GPU
        Returns detections as an Ultraytics object using the loaded model.
    """
    results = model(frame, device=device)
    return results

def train_model(cfg, data, epochs, imgsz, batch, name, verbose=True):
    """
        Train a detection model and create project in 
    """
    model = YOLO(cfg)
    
    results = model.train(data=data,
                              epochs=epochs, # Training Epochs...
                              imgsz=imgsz, # Image Size (e.g. 640 -> 640x640)
                              batch=batch, # Training Batch (Increases VRAM usage)
                              name=name, # project name
                              verbose=verbose # Disables prompt printing
                        )

def evaluate_model(weights_path):
    """Evaluate a single model and return its metrics."""
    model = load_model(weights_path)
    metrics = model.val(project=None, name=None, verbose=False)
    latency = np.sum(list(metrics.speed.values()))
    return {
        "Model": weights_path.split("/")[-1].replace(".pt", ""),
        "mAP50": float(metrics.results_dict["metrics/mAP50(B)"]),
        "mAP50-95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
        "recall": float(metrics.results_dict["metrics/recall(B)"]),
        "latency": float(latency)
    }
