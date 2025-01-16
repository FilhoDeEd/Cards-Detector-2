import os
import cv2
import numpy as np
import random

cv2.namedWindow('tela')

CARDS_PATH = 'datasets/cards'

# Carregar uma imagem de carta
cards = os.listdir(CARDS_PATH)
card_image_path = os.path.join(CARDS_PATH, cards[random.randint(0, len(cards) - 1)])

card_image = cv2.imread(card_image_path, cv2.IMREAD_UNCHANGED)  # Lendo com o canal alfa (transparência)

# Caso a imagem não tenha canal alfa (transparente), adicionar um
if card_image.shape[2] == 3:
    card_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2BGRA)

# Dimensões originais da carta
card_h, card_w, _ = card_image.shape

# Matriz de rotação - calculando o centro da imagem
matrix_rotation = cv2.getRotationMatrix2D((card_w / 2, card_h / 2), 50, 1)

# Calculando o tamanho novo da imagem para evitar corte
abs_cos = abs(matrix_rotation[0, 0])
abs_sin = abs(matrix_rotation[0, 1])

# Calcular a nova largura e altura
new_w = int(card_h * abs_sin + card_w * abs_cos)
new_h = int(card_h * abs_cos + card_w * abs_sin)

# Atualizar a matriz de rotação para garantir que a imagem centralize corretamente após rotação
matrix_rotation[0, 2] += (new_w / 2) - card_w / 2
matrix_rotation[1, 2] += (new_h / 2) - card_h / 2

# Realizar a rotação sem cortar e mantendo a transparência
rotated_image = cv2.warpAffine(card_image, matrix_rotation, (new_w, new_h))

# Criar o fundo amarelo (não transparente)
yellow_background = np.ones((new_h, new_w, 3), dtype=np.uint8) * 100  # Fundo amarelo (RGB)

# Colar a carta rotacionada no fundo amarelo, levando em conta a transparência
for y in range(rotated_image.shape[0]):
    for x in range(rotated_image.shape[1]):
        if rotated_image[y, x][3] != 0:  # Verifica se o pixel não é totalmente transparente
            yellow_background[y, x] = rotated_image[y, x][:3]  # Copia a cor RGB da carta

# Exibição da imagem final com fundo amarelo
while True:
    cv2.imshow('tela', yellow_background)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Aperte 'q' para sair
        break

cv2.destroyAllWindows()

# Salvar a imagem final com fundo amarelo
output_path = "rotated_card_with_yellow_background.png"
cv2.imwrite(output_path, yellow_background)
