import cv2
import gdown
import kagglehub
import math
import numpy as np
import os
import random
import zipfile

from collections import namedtuple
from cv2.typing import MatLike
from numpy.typing import NDArray
from time import sleep
from typing import List, Tuple

Range = namedtuple("Range", ["min", "max"])


cv2.namedWindow('cartas') #temp
cv2.namedWindow('backgroud') #temp

CARD_WIDTH = 200 # 250
CARD_HEIGHT = 290 # 363
CARD_DIAGONAL = math.hypot(CARD_WIDTH, CARD_HEIGHT)
BACKGROUND_SIZE = 960
CARDS_PATH = 'datasets/cards'

CARDS_RANGE = Range(0, 4)
ANGLE_RANGE = Range(-90, 90)
SHEAR_X_RANGE = Range(-0.1, 0.1)
SHEAR_Y_RANGE = Range(-0.1, 0.1)
SCALE_RANGE = Range(0.7, 1.2)
BLUR_RANGE = Range(3, 7)
TRANSLATE_STEP = 50
PREVENT_OVERLAPPING = True

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
    51: ['5_of_spades']
}


def read_card(card_name: str) -> MatLike:
    card_path = os.path.join(CARDS_PATH, f'{card_name}.png')
    card_image = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
    card_image = cv2.resize(card_image, (CARD_WIDTH, CARD_HEIGHT))
    transparent_background = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE, 4), dtype=np.uint8)
    card_pos_x = (BACKGROUND_SIZE - CARD_HEIGHT) // 2
    card_pos_y = (BACKGROUND_SIZE - CARD_WIDTH) // 2
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

    height, width, depth = image.shape

    new_width = int(width * scale)
    new_height = int(height * scale)

    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    crop_y = max((new_height - BACKGROUND_SIZE) // 2, 0)
    crop_x = max((new_width - BACKGROUND_SIZE) // 2, 0)

    cropped_scaled_image = scaled_image[
        crop_y:crop_y + min(new_height, BACKGROUND_SIZE),
        crop_x:crop_x + min(new_width, BACKGROUND_SIZE)
    ]

    background = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE, depth), dtype=image.dtype)

    start_y = max((BACKGROUND_SIZE - new_height) // 2, 0)
    start_x = max((BACKGROUND_SIZE - new_width) // 2, 0)

    background[
        start_y:start_y + cropped_scaled_image.shape[0],
        start_x:start_x + cropped_scaled_image.shape[1]
    ] = cropped_scaled_image

    return background


def blur_image(image: MatLike, kernel_size: int) -> MatLike:
    kernel_size = kernel_size
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return blurred_image


def translate_image(image: MatLike, dx: int, dy: int) -> MatLike:
    height, width, _ = image.shape

    translation_matrix = np.array([
        [1, 0, dx],
        [0, 1, dy]
    ], dtype=np.float32)

    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

    return translated_image


def get_random_translate(last_translates: List[Tuple], width: int, height: int, prevent_overlapping: bool = False) -> Tuple[int, int]:
    max_translate = int((BACKGROUND_SIZE - SCALE_RANGE.max * CARD_DIAGONAL) // 2)

    if prevent_overlapping:
        for _ in range(100):
            dx = random.randrange(-max_translate, max_translate, TRANSLATE_STEP)
            dy = random.randrange(-max_translate, max_translate, TRANSLATE_STEP)

            new_card = (dx, dy, width, height)
            if not any(is_overlapping(new_card, card) for card in last_translates):
                last_translates.append((dx, dy, width, height))
                return dx, dy
    else:
        dx = random.randrange(-max_translate, max_translate, TRANSLATE_STEP)
        dy = random.randrange(-max_translate, max_translate, TRANSLATE_STEP)
        last_translates.append((dx, dy, width, height))
        return dx, dy

    return 0, 0


def is_overlapping(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


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

    rotated_bounding_box = np.dot(bounding_box - bounding_box.mean(axis=0), rotation_matrix.T) + bounding_box.mean(axis=0)

    return rotated_bounding_box


def translate_bounding_box(bounding_box: NDArray, dx: int, dy: int) -> NDArray:
    translated_bounding_box = bounding_box + np.array([dx, dy])

    return translated_bounding_box


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


def overlay_images(background: MatLike, card: MatLike, dx: int, dy: int) -> MatLike:
    # Define the region of interest (ROI) on the background
    roi = background[dy:dy + card.shape[0], dx:dx + card.shape[1]]

    # Assuming the card image has an alpha channel, extract the mask (alpha channel)
    card_alpha = card[:, :, 3] if card.shape[2] == 4 else np.ones((card.shape[0], card.shape[1]), dtype=np.uint8)

    # Ensure the mask is uint8
    mask = card_alpha.astype(np.uint8)

    # Apply the mask to the ROI
    masked_roi = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    # Now combine the card image with the background using the mask
    card_rgb = card[:, :, :3]  # Discard the alpha channel for blending
    masked_card = cv2.bitwise_and(card_rgb, card_rgb, mask=mask)

    # Place the card on the background
    combined = cv2.add(masked_roi, masked_card)
    background[dy:dy + card.shape[0], dx:dx + card.shape[1]] = combined

    return background


def main() -> None:
    background_images = []

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
        gdown.download(url=google_drive_url, output=google_drive_zip, quiet=False, fuzzy=True)

    if not os.path.exists(white_images_path) and os.path.exists(google_drive_zip) and zipfile.is_zipfile(google_drive_zip):
        with zipfile.ZipFile(google_drive_zip, 'r') as zip_ref:
            zip_ref.extractall('datasets/')

    for root, _, files in os.walk(white_images_path, topdown=False):
        for name in files:
            if '.jpg' in name or '.jpeg' in name:
                white_images.append(os.path.join(root, name))

    background_images.extend(white_images)

    card_classes = list(CLASSES.keys())
    background_images_count = len(background_images)
    images_per_class = background_images_count // len(card_classes)

    images_per_class = ((CARDS_RANGE.min + CARDS_RANGE.max) // 2) * images_per_class
    remaining_images_by_class = {card_class: images_per_class for card_class in card_classes}

    random.shuffle(background_images)

    for background_image_path in background_images:
        try:
            background_image = cv2.imread(background_image_path)
            background_image = cv2.resize(background_image, (BACKGROUND_SIZE, BACKGROUND_SIZE))
        except Exception as e:
            print(f"Erro ao processar {background_image_path}: {e}")
            continue

        last_translates = []
        number_cards = random.randint(CARDS_RANGE.min, CARDS_RANGE.max)

        if sum(list(remaining_images_by_class.values())) < number_cards:
            break

        for _ in range(number_cards):
            print('a')

            angle = random.uniform(ANGLE_RANGE.min, ANGLE_RANGE.max)
            shear_x = random.uniform(SHEAR_X_RANGE.min, SHEAR_X_RANGE.max)
            shear_y = random.uniform(SHEAR_Y_RANGE.min, SHEAR_Y_RANGE.max)
            scale = random.uniform(SCALE_RANGE.min, SCALE_RANGE.max)
            blur = random.randrange(BLUR_RANGE.min, BLUR_RANGE.max, 2)
            dx, dy = get_random_translate(last_translates, scale*CARD_WIDTH, scale*CARD_HEIGHT, PREVENT_OVERLAPPING)

            remaining_classes = list(remaining_images_by_class.keys())
            random_card_class = random.choice(remaining_classes)
            remaining_images_by_class[random_card_class] -= 1
            if remaining_images_by_class[random_card_class] == 0:
                del remaining_images_by_class[random_card_class]

            cards_for_class = CLASSES[random_card_class]
            card_image = read_card(random.choice(cards_for_class))

            card_image = shear_image(card_image, shear_x, shear_y)

            card_image = scale_image(card_image, scale)

            bounding_box = find_bounding_box(card_image)

            card_image = rotate_image(card_image, angle)
            bounding_box = rotate_bounding_box(bounding_box, angle)

            card_image = translate_image(card_image, dx, dy)
            bounding_box = translate_bounding_box(bounding_box, dx, dy)

            card_image = blur_image(card_image, blur)

            card_image = draw_bounding_box(card_image, bounding_box)

            #background_image = overlay_images(background_image, card_image)

            # Colocar na imagem randomica

            # normalized_oriented_bounding_box = normalize_oriented_bounding_box_yolo(oriented_bounding_box, card_image.shape) # o shape será da imagem final
            # print(normalized_oriented_bounding_box)

            cv2.imshow('cards', card_image)
            #cv2.imshow('background', background_image)

            sleep(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
