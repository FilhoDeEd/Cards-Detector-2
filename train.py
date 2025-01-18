import os
from ultralytics import YOLO


MODEL_NAME = 'cards_detector'
YOLO_BASE_MODEL = 'yolov8n-obb.pt'
DATASET_YAML_PATH = 'datasets/cards_detector_42/data.yml'
IMAGE_SIZE = 960
BATCH = 4
EPOCHS = 50


def main() -> None:
    model = YOLO(YOLO_BASE_MODEL)
    model.train(
        data=os.path.abspath(DATASET_YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMAGE_SIZE,
        project='runs/train',
        name=MODEL_NAME
    )


if __name__ == '__main__':
    main()
