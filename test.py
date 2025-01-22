import cv2
import os

from time import sleep
from ultralytics import YOLO


model_path = 'models/best.pt'
test_folder = 'datasets/cards_detector_42/test/images'

cv2.namedWindow('tela')

model = YOLO(model_path)

test_images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
if not test_images:
    raise FileNotFoundError(f'Nenhuma imagem encontrada na pasta {test_folder}!')

for image_name in test_images:
    test_image_path = os.path.join(test_folder, image_name)

    frame = cv2.imread(test_image_path)

    results = model.predict(test_image_path)

    annotated_frame = results[0].plot()

    cv2.imshow('tela', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(4)

cv2.destroyAllWindows()
