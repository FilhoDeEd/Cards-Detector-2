import cv2
from time import sleep
from ultralytics import YOLO


model_path = 'models/best.pt'

cv2.namedWindow('tela')
cap = cv2.VideoCapture(0)

model = YOLO(model_path)

if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    exit()

def resize_and_crop_to_square(frame, size=960):
    h, w, _ = frame.shape

    scale = size / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - size) // 2
    start_y = (new_h - size) // 2
    cropped_frame = resized_frame[start_y:start_y + size, start_x:start_x + size]

    return cropped_frame

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar imagem!")
        break

    frame = resize_and_crop_to_square(frame)

    results = model.predict(frame)

    if results[0].obb and results[0].obb.data is not None:
        obb = results[0].obb

        max_index = obb.conf.argmax()
        best_xyxy = obb.xyxy[max_index]
        best_cls = int(obb.cls[max_index])
        best_conf = obb.conf[max_index].item()

        label = results[0].names[best_cls].replace("_", " ")
        text = f'{label} {best_conf:.2f}'

        x1, y1, x2, y2 = map(int, best_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20

        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                      (text_x + text_size[0], text_y + 5), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness
        )
    else:
        print("Nenhuma predição encontrada no atributo 'obb'.")

    cv2.imshow('tela', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(0.1)

cap.release()
cv2.destroyAllWindows()
