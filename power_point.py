import cv2
from create_dataset import (
    read_card,
    shear_image,
    scale_image,
    find_bounding_boxes,
    rotate_image,
    rotate_bounding_box,
    translate_image,
    translate_bounding_box,
    blur_image,
    change_image_brightness,
    overlay_images,
    add_noise_to_image,
    draw_bounding_box
)

# Constantes de transformação
CARD_NAME = 'ace_of_spades'
SHEAR_X = 0.05
SHEAR_Y = 0.03
SCALE_FACTOR = 1.5
ROTATION_ANGLE = 15
TRANSLATE_DX = 100
TRANSLATE_DY = -50
BLUR_KERNEL_SIZE = 7
BRIGHTNESS_FACTOR = 0.6
NOISE_TYPE = 'salt_and_pepper'
NOISE_INTENSITY = 0.03
BACKGROUND_IMAGE_PATH = 'background.jpeg'

cv2.namedWindow('Demo')

# Ler a carta
card = read_card(card_name=CARD_NAME)
cv2.imshow('Demo', card)
cv2.waitKey(0)

# Aplicar cisalhamento
shear = shear_image(image=card, shear_x=SHEAR_X, shear_y=SHEAR_Y)
cv2.imshow('Demo', shear)
cv2.waitKey(0)

# Escalar a imagem
scale = scale_image(image=shear, scale=SCALE_FACTOR)
cv2.imshow('Demo', scale)
cv2.waitKey(0)

# Encontrar e desenhar bounding boxes
obb1, obb2 = find_bounding_boxes(image=scale)
temp = draw_bounding_box(image=scale, bounding_box=obb1)
temp = draw_bounding_box(image=temp, bounding_box=obb2)
cv2.imshow('Demo', temp)
cv2.waitKey(0)

# Rotacionar a imagem
rotate = rotate_image(image=scale, angle=ROTATION_ANGLE)
temp = draw_bounding_box(image=rotate, bounding_box=obb1)
temp = draw_bounding_box(image=temp, bounding_box=obb2)
cv2.imshow('Demo', temp)
cv2.waitKey(0)

# Ajustar bounding boxes para a rotação
height, width, _ = rotate.shape
center = [width // 2, height // 2]
obb1 = rotate_bounding_box(bounding_box=obb1, angle=ROTATION_ANGLE, center=center)
obb2 = rotate_bounding_box(bounding_box=obb2, angle=ROTATION_ANGLE, center=center)
temp = draw_bounding_box(image=rotate, bounding_box=obb1)
temp = draw_bounding_box(image=temp, bounding_box=obb2)
cv2.imshow('Demo', temp)
cv2.waitKey(0)

# Transladar a imagem
translate = translate_image(image=rotate, dx=TRANSLATE_DX, dy=TRANSLATE_DY)
temp = draw_bounding_box(image=translate, bounding_box=obb1)
temp = draw_bounding_box(image=temp, bounding_box=obb2)
cv2.imshow('Demo', temp)
cv2.waitKey(0)

# Ajustar bounding boxes para a translação
obb1 = translate_bounding_box(bounding_box=obb1, dx=TRANSLATE_DX, dy=TRANSLATE_DY)
obb2 = translate_bounding_box(bounding_box=obb2, dx=TRANSLATE_DX, dy=TRANSLATE_DY)
temp = draw_bounding_box(image=translate, bounding_box=obb1)
temp = draw_bounding_box(image=temp, bounding_box=obb2)
cv2.imshow('Demo', temp)
cv2.waitKey(0)

# Aplicar desfoque
blur = blur_image(image=temp, kernel_size=BLUR_KERNEL_SIZE)
cv2.imshow('Demo', blur)
cv2.waitKey(0)

# Ajustar brilho
brightness = change_image_brightness(image=blur, brightness_factor=BRIGHTNESS_FACTOR)
cv2.imshow('Demo', brightness)
cv2.waitKey(0)

# Carregar e sobrepor com o plano de fundo
background = cv2.imread(BACKGROUND_IMAGE_PATH)
overlay = overlay_images(background_image=background, card_image=brightness)
cv2.imshow('Demo', overlay)
cv2.waitKey(0)

# Adicionar ruído
noise = add_noise_to_image(image=overlay, noise_type=NOISE_TYPE, intensity=NOISE_INTENSITY)
cv2.imshow('Demo', noise)
cv2.waitKey(0)

cv2.destroyAllWindows()
