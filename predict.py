import cv2
from time import sleep
from ultralytics import YOLO

model_path = 'models/best.pt'

cv2.namedWindow('tela')

cap = cv2.VideoCapture(0)

# Inicializa o modelo YOLO
model = YOLO(model_path)

if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    exit()

def resize_and_crop_to_square(frame, size=960):
    """Redimensiona e corta o frame para garantir um quadrado de dimensões específicas."""
    h, w, _ = frame.shape

    # Calcula fator de escala
    scale = size / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Redimensiona para que a menor dimensão atinja 'size'
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Corta para o centro para ajustar para (size x size)
    start_x = (new_w - size) // 2
    start_y = (new_h - size) // 2
    cropped_frame = resized_frame[start_y:start_y + size, start_x:start_x + size]

    return cropped_frame

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar imagem!")
        break

    # Redimensiona e corta o frame para 960x960
    frame = resize_and_crop_to_square(frame)

    # Inferência com o YOLO
    results = model.predict(frame)
    annotated_frame = results[0].plot()

    # Exibe o resultado na janela
    cv2.imshow('tela', annotated_frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleep(0.1)

cap.release()
cv2.destroyAllWindows()
