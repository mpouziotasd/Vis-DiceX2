import argparse
from ultralytics import YOLO

def begin_train(cfg, data, epochs, imgsz, batch, name, verbose=True):
    model = YOLO(cfg)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model.")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the YOLOv8 configuration file (e.g., yolov8-M.yaml).")
    parser.add_argument('--data', type=str, default="data/Vis-DiceX2.yaml", help="Path to the dataset YAML file.")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs.")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training.")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--name', type=str, default='MyTrain', help="Name of the training session.")

    args = parser.parse_args()
    begin_train(
        cfg=args.cfg,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name
    )
