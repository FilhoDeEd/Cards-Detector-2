import cv2
from create_dataset import (
    read_card,
    rotate_image,
    shear_image,
    scale_image,
    blur_image,
    add_noise_to_image,
    change_image_brightness,
    draw_bounding_box,
    find_bounding_boxes,
)

# Carregar uma imagem de exemplo
card_name = "ace_of_spades"  # Substitua pelo nome da carta que você tem no dataset
image = read_card(card_name)

# Lista de transformações a serem aplicadas sequencialmente
transformations = [
    ("Rotate", lambda img: rotate_image(img, angle=15)),
    ("Shear", lambda img: shear_image(img, shear_x=0.05, shear_y=0.03)),
    ("Scale", lambda img: scale_image(img, scale=1.2)),
    ("Blur", lambda img: blur_image(img, 5)),
    ("Add Noise", lambda img: add_noise_to_image(img, noise_type="gaussian", intensity=0.08)),
    ("Brightness", lambda img: change_image_brightness(img, brightness_factor=0.6)),
]

# Aplicar as transformações sequencialmente
current_image = image
for name, transform in transformations:
    # Aplicar a transformação na imagem atual
    current_image = transform(current_image)
    
    # Exibir o resultado da transformação
    cv2.imshow(name, current_image)
    print(f"Exibindo: {name}. Pressione qualquer tecla para avançar...")
    cv2.waitKey(0)

# Fechar todas as janelas ao finalizar
cv2.destroyAllWindows()
