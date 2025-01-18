import cv2
from time import sleep
from ultralytics import YOLO

model_path = 'runs/train/cards_detector2/weights/best.pt'

cv2.namedWindow('tela')

model = YOLO(model_path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar imagem!")
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()

    cv2.imshow('tela', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(0.1)

cap.release()
cv2.destroyAllWindows()
