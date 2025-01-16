import os
import kagglehub
import cv2
import numpy as np
import random

from cv2.typing import MatLike

cv2.namedWindow('tela') #temp

CARD_WIDTH = 250
CARD_HEIGHT = 363
BACKGROUD = 450

CARDS_PATH = 'datasets/cards'

CARDS = os.listdir(CARDS_PATH)

CLASSES = {
    '2_of_hearts': 0,
    '6_of_hearts': 1,
    'jack_of_spades2': 2,
    '10_of_diamonds': 3,
    '5_of_spades': 4,
    'king_of_diamonds2': 5,
    'jack_of_diamonds2': 6,
    '8_of_diamonds': 7,
    'queen_of_clubs2': 8,
    '3_of_clubs': 9,
    'jack_of_clubs2': 10,
    'king_of_clubs2': 11,
    '3_of_diamonds': 12,
    'ace_of_spades': 13,
    'ace_of_spades2': 13,
    '6_of_spades': 14,
    '7_of_spades': 15,
    '7_of_hearts': 16,
    '3_of_spades': 17,
    '10_of_hearts': 18,
    '4_of_hearts': 19,
    '8_of_spades': 20,
    '9_of_hearts': 21,
    'jack_of_hearts2': 22,
    '6_of_diamonds': 23,
    '8_of_clubs': 24,
    'queen_of_diamonds2': 25,
    '2_of_clubs': 26,
    '10_of_spades': 27,
    '4_of_spades': 28,
    'ace_of_hearts': 29,
    '7_of_clubs': 30,
    '4_of_clubs': 31,
    '9_of_diamonds': 32,
    '10_of_clubs': 33,
    'queen_of_hearts2': 34,
    '5_of_clubs': 35,
    'king_of_hearts2': 36,
    '5_of_hearts': 37,
    '7_of_diamonds': 38,
    '3_of_hearts': 39,
    '9_of_clubs': 40,
    '6_of_clubs': 41,
    'queen_of_spades2': 42,
    'ace_of_diamonds': 43,
    '8_of_hearts': 44,
    'king_of_spades2': 45,
    '5_of_diamonds': 46,
    '2_of_spades': 47,
    '2_of_diamonds': 48,
    '4_of_diamonds': 49,
    '9_of_spades': 50,
    'ace_of_clubs': 51
}


def read_card(card_path: str) -> MatLike:
    card_image = cv2.imread(os.path.join(CARDS_PATH, card_path), cv2.IMREAD_UNCHANGED)
    card_image = cv2.resize(card_image, (CARD_WIDTH, CARD_HEIGHT))
    transparent_background = np.zeros((BACKGROUD, BACKGROUD, 4), dtype=np.uint8)
    card_pos_x = (BACKGROUD - CARD_HEIGHT) // 2
    card_pos_y = (BACKGROUD - CARD_WIDTH) // 2
    transparent_background[card_pos_x:card_pos_x + CARD_HEIGHT, card_pos_y:card_pos_y + CARD_WIDTH] = card_image

    return transparent_background

def rotate_image(image: MatLike, angle: int) -> MatLike:
    height, width, _ = image.shape
    matrix_rotation = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
    return cv2.warpAffine(image, matrix_rotation, (height, width))

def shear_image(image: MatLike, shear_x: float, shear_y: float) -> MatLike:
    height, width, _ = image.shape

    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ], dtype=np.float32)

    new_width = int(width + abs(shear_x) * width)
    new_height = int(height + abs(shear_y) * height)

    sheared_image = cv2.warpAffine(image, shear_matrix, (new_width, new_height))

    crop_y = (new_height - height) // 2
    crop_x = (new_width - width) // 2

    return sheared_image[crop_y: new_height - crop_y, crop_x: new_width - crop_x]

path = kagglehub.dataset_download('ezzzio/random-images')

card_image = read_card(CARDS[random.randint(0, len(CARDS) - 1)])

angle = 30
card_image = rotate_image(card_image, angle)

shear_x = 0.0
shear_y = 0.3
card_image = shear_image(card_image, shear_x, shear_y)

while True:
    cv2.imshow('tela', card_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Aperte 'q' para sair
        break

cv2.destroyAllWindows()

cv2.imwrite('image.png', card_image)

random_images = []
for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
      if '.jpg' in name:
        random_images.append(os.path.join(root, name))

#print(random_images)
