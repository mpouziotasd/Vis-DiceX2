import argparse
from ultralytics import YOLO

def begin_evaluation(cfg, data, batch, conf, iou, name, verbose=True):
    model = YOLO(cfg)
    model.val(
        data=data,
        batch=batch,
        conf=conf,
        iou=iou,
        name=name
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model.")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the YOLOv8 configuration file (e.g., yolov8-M.yaml).")
    parser.add_argument('--data', type=str, default="data/Vis-DiceX2.yaml", help="Path to the dataset YAML file.")
    parser.add_argument('--conf', type=int, default=0.001, help="Select Model Confidence for evaluation (Default=0.001) Can be set from 0.001 to 0.99-")
    parser.add_argument('--iou', type=int, default=0.6, help="Select Model IoU for evaluation (Default=0.6). Can be set from 0 to 0.95")
    parser.add_argument('--batch', type=int, default=-1, help="Batch size for training (Default=-1 Automatically adjusts batch size based on GPU memory)")
    parser.add_argument('--name', type=str, default='MyTrain', help="Name of the evaluation session.")
    parser.add_argument('--name', type=str, default='MyTrain', help="Name of the evaluation session.")

    
    args = parser.parse_args()
    begin_evaluation(
        cfg=args.cfg,
        data=args.data,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        name=args.name
    )
