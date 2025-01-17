import os
import cv2
import gdown
import kagglehub
import numpy as np
import random
import zipfile

from cv2.typing import MatLike
from numpy.typing import NDArray
from time import sleep

from pprint import pprint

cv2.namedWindow('tela') #temp

CARD_WIDTH = 250
CARD_HEIGHT = 363
BACKGROUD = 500

CARDS_PATH = 'datasets/cards'

CARDS = os.listdir(CARDS_PATH)

CLASSES = {
    0: ['6_of_spades'],
    1: ['8_of_diamonds'],
    2: ['2_of_hearts'],
    3: ['ace_of_diamonds'],
    4: ['king_of_clubs2'],
    5: ['3_of_clubs'],
    6: ['4_of_clubs'],
    7: ['jack_of_clubs2'],
    8: ['5_of_hearts'],
    9: ['3_of_hearts'],
    10: ['ace_of_clubs'],
    11: ['4_of_hearts'],
    12: ['ace_of_spades', 'ace_of_spades2'],
    13: ['queen_of_clubs2'],
    14: ['2_of_spades'],
    15: ['king_of_spades2'],
    16: ['2_of_diamonds'],
    17: ['jack_of_diamonds2'],
    18: ['king_of_diamonds2'],
    19: ['5_of_clubs'],
    20: ['king_of_hearts2'],
    21: ['3_of_diamonds'],
    22: ['9_of_diamonds'],
    23: ['ace_of_hearts'],
    24: ['6_of_clubs'],
    25: ['4_of_diamonds'],
    26: ['jack_of_hearts2'],
    27: ['queen_of_spades2'],
    28: ['9_of_clubs'],
    29: ['5_of_diamonds'],
    30: ['9_of_hearts'],
    31: ['7_of_spades'],
    32: ['queen_of_diamonds2'],
    33: ['8_of_hearts'],
    34: ['8_of_clubs'],
    35: ['6_of_diamonds'],
    36: ['9_of_spades'],
    37: ['8_of_spades'],
    38: ['7_of_diamonds'],
    39: ['6_of_hearts'],
    40: ['jack_of_spades2'],
    41: ['4_of_spades'],
    42: ['3_of_spades'],
    43: ['10_of_hearts'],
    44: ['7_of_hearts'],
    45: ['2_of_clubs'],
    46: ['queen_of_hearts2'],
    47: ['10_of_diamonds'],
    48: ['10_of_clubs'],
    49: ['7_of_clubs'],
    50: ['10_of_spades'],
    51: ['5_of_spadeg']
}


def read_card(card_path: str) -> MatLike:
    card_image = cv2.imread(os.path.join(CARDS_PATH, card_path), cv2.IMREAD_UNCHANGED)
    card_image = cv2.resize(card_image, (CARD_WIDTH, CARD_HEIGHT))
    transparent_background = np.zeros((BACKGROUD, BACKGROUD, 4), dtype=np.uint8)
    card_pos_x = (BACKGROUD - CARD_HEIGHT) // 2
    card_pos_y = (BACKGROUD - CARD_WIDTH) // 2
    transparent_background[card_pos_x:card_pos_x + CARD_HEIGHT, card_pos_y:card_pos_y + CARD_WIDTH] = card_image

    return transparent_background

def rotate_image(image: MatLike, angle: float) -> MatLike:
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    matrix_rotation = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, matrix_rotation, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated_image

def shear_image(image: MatLike, shear_x: float, shear_y: float) -> MatLike:
    height, width, _ = image.shape

    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ], dtype=np.float32)

    dx = abs(shear_x * height)
    dy = abs(shear_y * width)

    new_width = int(width + dx)
    new_height = int(height + dy)

    shear_matrix[0, 2] = max(-shear_x * height, 0)
    shear_matrix[1, 2] = max(-shear_y * width, 0)

    sheared_image = cv2.warpAffine(image, shear_matrix, (new_width, new_height))

    start_x = max(0, int((new_width - width) // 2))
    start_y = max(0, int((new_height - height) // 2))

    return sheared_image[start_y: start_y + height, start_x: start_x + width]

def scale_image(image: MatLike, scale: float) -> MatLike:
    if scale <= 0:
        raise ValueError("O fator de escala deve ser maior que 0.")

    height, width, _ = image.shape

    new_width = int(width * scale)
    new_height = int(height * scale)

    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return scaled_image

def blur_image(image: MatLike, kernel_size: int) -> MatLike:
    kernel_size = kernel_size
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return blurred_image

def find_bounding_box(image: MatLike) -> NDArray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])

        return np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ], dtype=np.float32)

def rotate_bounding_box(bounding_box: NDArray, angle: float) -> NDArray:
        angle_rad = np.deg2rad(angle)

        rotation_matrix = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=np.float32)

        rotated_rect = np.dot(bounding_box - bounding_box.mean(axis=0), rotation_matrix.T) + bounding_box.mean(axis=0)

        return rotated_rect

def normalize_oriented_bounding_box_yolo(bounding_box: NDArray, image_shape: tuple) -> NDArray:
    image_height, image_width, _ = image_shape
    
    points = bounding_box.flatten()

    normalized_points = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]
        normalized_points.append(x / image_width)
        normalized_points.append(y / image_height)

    return np.array(normalized_points)

def draw_bounding_box(image: MatLike, bounding_box: NDArray) -> NDArray:
    bounding_box = np.round(bounding_box).astype(int)

    for i in range(4):
        cv2.line(image, tuple(bounding_box[i]), tuple(bounding_box[(i+1) % 4]), (0, 255, 0), 2)

    return image


def main() -> None:
    background_images = []
    classes = list(CLASSES.keys())

    random_images = []
    kaggle_path = 'felicepollano/watermarked-not-watermarked-images'
    random_images_path = kagglehub.dataset_download(kaggle_path)
    for root, _, files in os.walk(random_images_path, topdown=False):
        for name in files:
            if '.jpg' in name or '.jpeg' in name:
                random_images.append(os.path.join(root, name))

    background_images.extend(random.sample(random_images, 7000))

    white_images = []
    google_drive_zip = 'datasets/white_images.zip'
    white_images_path = 'datasets/white_images'
    if not os.path.exists(google_drive_zip):
        google_drive_url = 'https://drive.google.com/file/d/1gQFYum5XkJRmLWyMKjM_QvepTdL8Lwh8/view?usp=drive_link'
        gdown.download(url=google_drive_url, output=google_drive_zip, quiet=False,fuzzy=True)

    if not os.path.exists(white_images_path) and os.path.exists(google_drive_zip) and zipfile.is_zipfile(google_drive_zip):
        with zipfile.ZipFile(google_drive_zip, 'r') as zip_ref:
            zip_ref.extractall('datasets/')

    for root, _, files in os.walk(white_images_path, topdown=False):
        for name in files:
            if '.jpg' in name or '.jpeg' in name:
                white_images.append(os.path.join(root, name))

    background_images.extend(white_images)

    background_images_count = len(background_images)
    print(background_images_count)
    images_per_class = background_images_count//len(classes)

    remaining_images_by_class = {class_name: 2*images_per_class for class_name in classes}

    random.shuffle(background_images)

    for background_image_path in background_images:
        try:
            background_image = cv2.imread(background_image_path)
            background_image = cv2.resize(background_image, (640, 640))
        except Exception as e:
            print(f"Erro ao processar {background_image_path}: {e}")
            continue

        # angle = random.uniform(-90, 90)
        # shear_x = random.uniform(-0.1, 0.1)
        # shear_y = random.uniform(-0.1, 0.1)
        # scale = random.uniform(0.7, 1.3)
        # blur = random.randint(3, 10)

        # card_image = read_card(CARDS[random.randint(0, len(CARDS) - 1)])
        # card_image = shear_image(card_image, shear_x, shear_y)
        # card_image = scale_image(card_image, scale)

        # bounding_box = find_bounding_box(card_image)

        # card_image = rotate_image(card_image, angle)
        # oriented_bounding_box = rotate_bounding_box(bounding_box, angle)

        # card_image = draw_bounding_box(card_image, oriented_bounding_box)

        # card_image = blur_image(card_image, blur)

        # # Colocar na imagem randomica

        # # normalized_oriented_bounding_box = normalize_oriented_bounding_box_yolo(oriented_bounding_box, card_image.shape) # o shape será da imagem final
        # # print(normalized_oriented_bounding_box)

        cv2.imshow('tela', background_image)

        sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
