import cv2
import gdown
import hashlib
import kagglehub
import math
import numpy as np
import os
import random
import zipfile

from collections import namedtuple
from cv2.typing import MatLike
from numpy.typing import NDArray
from typing import Any, List, Tuple


Range = namedtuple('Range', ['min', 'max'])


SEED = 1024
random.seed(SEED)

NEW_DATASET_PATH = f'datasets/cards_detector_{SEED}'
TRAIN_SPLIT = 0.5
TEST_SPLIT = 0.3
VALIDATION_SPLIT = 0.2
MAX_DATASET_BATCH = 500
RANDOM_IMAGES_AMOUNT = 4000 #4984     # Until 31368
WHITE_IMAGES_AMOUNT = 1000 #2016      # Until 2016
NO_LABELS_AMOUNT = 500

CARD_WIDTH = 200                      # Other 250
CARD_HEIGHT = 290                     # Other 363
CARD_DIAGONAL = math.hypot(CARD_WIDTH, CARD_HEIGHT)
BACKGROUND_SIZE = 960
CARDS_PATH = 'datasets/cards'

CARDS_RANGE = Range(1, 2)
ANGLE_RANGE = Range(-15, 15)
SHEAR_X_RANGE = Range(-0.06, 0.06)
SHEAR_Y_RANGE = Range(-0.06, 0.06)
SCALE_RANGE = Range(0.7, 1.5)
BLUR_RANGE = Range(3, 7)
NOISE_RANGE = Range(0.02, 0.1)
BRIGHTNESS_RANGE = Range(0.4, 0.8)

TRANSLATE_STEP = 50
PREVENT_OVERLAPPING = True

CLASSES = {
    0: ['6_of_spades', '6_of_spades2'],
    1: ['8_of_diamonds', '8_of_diamonds2'],
    2: ['2_of_hearts', '2_of_hearts2'],
    3: ['ace_of_diamonds', 'ace_of_diamonds2'],
    4: ['king_of_clubs', 'king_of_clubs2'],
    5: ['3_of_clubs', '3_of_clubs2'],
    6: ['4_of_clubs', '4_of_clubs2'],
    7: ['jack_of_clubs', 'jack_of_clubs2'],
    8: ['5_of_hearts', '5_of_hearts2'],
    9: ['3_of_hearts', '3_of_hearts2'],
    10: ['ace_of_clubs', 'ace_of_clubs2'],
    11: ['4_of_hearts', '4_of_hearts2'],
    12: ['ace_of_spades', 'ace_of_spades2', 'ace_of_spades3'],
    13: ['queen_of_clubs', 'queen_of_clubs2'],
    14: ['2_of_spades', '2_of_spades2'],
    15: ['king_of_spades', 'king_of_spades2'],
    16: ['2_of_diamonds', '2_of_diamonds2'],
    17: ['jack_of_diamonds', 'jack_of_diamonds2'],
    18: ['king_of_diamonds', 'king_of_diamonds2'],
    19: ['5_of_clubs', '5_of_clubs2'],
    20: ['king_of_hearts', 'king_of_hearts2'],
    21: ['3_of_diamonds', '3_of_diamonds2'],
    22: ['9_of_diamonds', '9_of_diamonds2'],
    23: ['ace_of_hearts', 'ace_of_hearts2'],
    24: ['6_of_clubs', '6_of_clubs2'],
    25: ['4_of_diamonds', '4_of_diamonds2'],
    26: ['jack_of_hearts', 'jack_of_hearts2'],
    27: ['queen_of_spades', 'queen_of_spades2'],
    28: ['9_of_clubs', '9_of_clubs2'],
    29: ['5_of_diamonds', '5_of_diamonds2'],
    30: ['9_of_hearts', '9_of_hearts2'],
    31: ['7_of_spades', '7_of_spades2'],
    32: ['queen_of_diamonds', 'queen_of_diamonds2'],
    33: ['8_of_hearts', '8_of_hearts2'],
    34: ['8_of_clubs', '8_of_clubs2'],
    35: ['6_of_diamonds', '6_of_diamonds2'],
    36: ['9_of_spades', '9_of_spades2'],
    37: ['8_of_spades', '8_of_spades2'],
    38: ['7_of_diamonds', '7_of_diamonds2'],
    39: ['6_of_hearts', '6_of_hearts2'],
    40: ['jack_of_spades', 'jack_of_spades2'],
    41: ['4_of_spades', '4_of_spades2'],
    42: ['3_of_spades', '3_of_spades2'],
    43: ['10_of_hearts', '10_of_hearts2'],
    44: ['7_of_hearts', '7_of_hearts2'],
    45: ['2_of_clubs', '2_of_clubs2'],
    46: ['queen_of_hearts', 'queen_of_hearts2'],
    47: ['10_of_diamonds', '10_of_diamonds2'],
    48: ['10_of_clubs', '10_of_clubs2'],
    49: ['7_of_clubs', '7_of_clubs2'],
    50: ['10_of_spades', '10_of_spades2'],
    51: ['5_of_spades', '5_of_spades2']
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
        raise ValueError('O fator de escala deve ser maior que 0.')

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


def add_noise_to_image(image: MatLike, noise_type: str = "gaussian", intensity: float = 0.1) -> MatLike:
    if noise_type == "gaussian":
        mean = 0
        stddev = intensity * 255
        gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    elif noise_type == "salt_and_pepper":
        noisy_image = image.copy()
        salt_pepper_ratio = 0.5
        total_pixels = int(intensity * image.size)
        num_salt = int(salt_pepper_ratio * total_pixels)
        num_pepper = total_pixels - num_salt

        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 255

        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 0
    else:
        raise ValueError(f"Tipo de ruído '{noise_type}' não suportado.")

    return noisy_image


def change_image_brightness(image: MatLike, brightness_factor: float) -> MatLike:
    if image.shape[-1] == 4:
        rgb = image[..., :3].astype(np.float32)
        alpha = image[..., 3]
        
        rgb = rgb * brightness_factor
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        return np.dstack((rgb, alpha))
    else:
        image = image.astype(np.float32)
        image = image * brightness_factor
        return np.clip(image, 0, 255).astype(np.uint8)



def find_bounding_boxes(image: NDArray) -> tuple[NDArray, NDArray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])

        sub_width = w * 0.22
        sub_height = h * 0.27

        top_left_bbox = np.array([
            [x, y],
            [x + sub_width, y],
            [x + sub_width, y + sub_height],
            [x, y + sub_height],
        ], dtype=np.float32)

        bottom_right_bbox = np.array([
            [x + w - sub_width, y + h - sub_height],
            [x + w, y + h - sub_height],
            [x + w, y + h],
            [x + w - sub_width, y + h],
        ], dtype=np.float32)

        return top_left_bbox, bottom_right_bbox

    return None, None


def rotate_bounding_box(bounding_box: NDArray, angle: float, center: List[float]) -> NDArray:
    angle_rad = np.deg2rad(angle)

    rotation_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ], dtype=np.float32)

    center = np.array(center, dtype=np.float32)
    rotated_bounding_box = np.dot(bounding_box - center, rotation_matrix.T) + center

    return rotated_bounding_box


def translate_bounding_box(bounding_box: NDArray, dx: int, dy: int) -> NDArray:
    translated_bounding_box = bounding_box + np.array([dx, dy])

    return translated_bounding_box


def create_yolo_OBB(class_index: int, oriented_bounding_box: NDArray, image_shape: tuple) -> NDArray:
    image_height, image_width, _ = image_shape
    
    points = oriented_bounding_box.flatten()

    yolo_oriented_bounding_box = [class_index]
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]
        yolo_oriented_bounding_box.append(x / image_width)
        yolo_oriented_bounding_box.append(y / image_height)

    return np.array(yolo_oriented_bounding_box)


def draw_bounding_box(image: MatLike, bounding_box: NDArray) -> NDArray:
    bounding_box = np.round(bounding_box).astype(int)

    for i in range(4):
        cv2.line(image, tuple(bounding_box[i]), tuple(bounding_box[(i+1) % 4]), (0, 255, 0), 2)

    return image


def chance(probability: float) -> bool:
    return random.random() < probability


def overlay_images(background_image, card_image): 
    card_alpha = card_image[:, :, 3] / 255.0

    masked_background = background_image * (1 - card_alpha[:, :, None])
    masked_card = card_image * card_alpha[:, :, None]

    result = cv2.add(masked_background.astype(np.uint8), masked_card.astype(np.uint8))
    
    return result


def write_dataset(unique_info: (str | int), dataset_batch: Any, train_split: float, test_split: float) -> None:
    print('SAVING BATCH...')

    base_dirs = {
        'train': os.path.join(NEW_DATASET_PATH, 'train'),
        'test': os.path.join(NEW_DATASET_PATH, 'test'),
        'validation': os.path.join(NEW_DATASET_PATH, 'validation'),
    }

    for split in base_dirs:
        os.makedirs(os.path.join(base_dirs[split], 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dirs[split], 'labels'), exist_ok=True)

    total = len(dataset_batch)
    train_size = int(total * train_split)
    test_size = int(total * test_split)
    validation_size = total - train_size - test_size

    splits = (
        ['train'] * train_size +
        ['test'] * test_size +
        ['validation'] * validation_size
    )
    random.shuffle(splits)

    for i, (image, obbs) in enumerate(dataset_batch):
        split = splits[i]

        hash = hashlib.md5(f'{unique_info}_{SEED}_{i}'.encode()).hexdigest()[:8]
        image_filename = os.path.join(base_dirs[split], 'images', f'{hash}.png')
        bbox_filename = os.path.join(base_dirs[split], 'labels', f'{hash}.txt')

        cv2.imwrite(image_filename, image)

        with open(bbox_filename, 'w') as bbox_file:
            for obb in obbs:
                bbox_str = ' '.join(map(str, obb))
                bbox_file.write(f'{bbox_str}\n')


def main() -> None:
    background_images = []

    random_images = []
    kaggle_path = 'felicepollano/watermarked-not-watermarked-images'
    random_images_path = kagglehub.dataset_download(kaggle_path)
    for root, _, files in os.walk(random_images_path, topdown=False):
        for name in files:
            if '.jpg' in name or '.jpeg' in name:
                random_images.append(os.path.join(root, name))

    background_images.extend(random.sample(random_images, RANDOM_IMAGES_AMOUNT))

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

    background_images.extend(random.sample(white_images, WHITE_IMAGES_AMOUNT))

    card_classes = list(CLASSES.keys())
    background_images_count = len(background_images)
    images_per_class = background_images_count // len(card_classes)

    if (CARDS_RANGE.max - CARDS_RANGE.min) > 1:
        images_per_class = ((CARDS_RANGE.min + CARDS_RANGE.max) // 2) * images_per_class
    remaining_images_by_class = {card_class: images_per_class for card_class in card_classes}

    random.shuffle(background_images)

    dataset_batch = []

    progress = 0
    TOTAL = len(background_images)

    for background_image_path in background_images:
        print(f'PROGRESS: {progress}/{TOTAL}')

        try:
            background_image = cv2.imread(background_image_path)
            background_image = cv2.resize(background_image, (BACKGROUND_SIZE, BACKGROUND_SIZE))
            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2BGRA)
        except Exception as e:
            print(f'Erro ao processar {background_image_path}: {e}')
            continue

        last_translates = []
        background_image_OBBs = []
        background_image_filename = os.path.basename(background_image_path)
        if chance(NO_LABELS_AMOUNT/TOTAL):
            number_cards = 0
        else:
            number_cards = random.randint(CARDS_RANGE.min, CARDS_RANGE.max)

        if sum(list(remaining_images_by_class.values())) < number_cards:
            break

        for _ in range(number_cards):
            angle = random.uniform(ANGLE_RANGE.min, ANGLE_RANGE.max)
            shear_x = random.uniform(SHEAR_X_RANGE.min, SHEAR_X_RANGE.max)
            shear_y = random.uniform(SHEAR_Y_RANGE.min, SHEAR_Y_RANGE.max)
            scale = random.uniform(SCALE_RANGE.min, SCALE_RANGE.max)
            blur = random.randrange(BLUR_RANGE.min, BLUR_RANGE.max, 2)
            brightness = random.uniform(BRIGHTNESS_RANGE.min, BRIGHTNESS_RANGE.max)
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

            top_left_bbox, bottom_right_bbox = find_bounding_boxes(card_image)

            card_image = rotate_image(card_image, angle)
            height, width, _ = card_image.shape
            center = [width // 2, height // 2]
            top_left_bbox = rotate_bounding_box(top_left_bbox, angle, center)
            bottom_right_bbox = rotate_bounding_box(bottom_right_bbox, angle, center)

            card_image = translate_image(card_image, dx, dy)
            top_left_bbox = translate_bounding_box(top_left_bbox, dx, dy)
            bottom_right_bbox = translate_bounding_box(bottom_right_bbox, dx, dy)

            card_image = blur_image(card_image, blur)

            card_image = change_image_brightness(card_image, brightness)

            background_image = overlay_images(background_image, card_image)

            background_image = draw_bounding_box(background_image, top_left_bbox)
            background_image = draw_bounding_box(background_image, bottom_right_bbox)

            top_left_yolo_OBB = create_yolo_OBB(random_card_class, top_left_bbox, background_image.shape)
            bottom_right_yolo_OBB = create_yolo_OBB(random_card_class, bottom_right_bbox, background_image.shape)

            background_image_OBBs.append(top_left_yolo_OBB)
            background_image_OBBs.append(bottom_right_yolo_OBB)

        noise_type = random.choice(["gaussian", "salt_and_pepper"])
        noise_intensity = random.uniform(NOISE_RANGE.min, NOISE_RANGE.max)

        background_image = add_noise_to_image(background_image, noise_type, noise_intensity)

        dataset_batch.append((background_image, background_image_OBBs))

        if len(dataset_batch) >= MAX_DATASET_BATCH:
            write_dataset(background_image_filename, dataset_batch, TRAIN_SPLIT, TEST_SPLIT)
            dataset_batch = []

        progress += 1

    if len(dataset_batch) > 0:
        write_dataset(background_image_filename, dataset_batch, TRAIN_SPLIT, TEST_SPLIT)
        dataset_batch = []


if __name__ == '__main__':
    main()
