# Detector de Cartas

## Sistema de Detecção de Cartas de Baralho com YOLO

Este projeto utiliza **Inteligência Artificial (IA)** para identificar cartas de baralho padrão (52 cartas) em tempo real, capturadas por uma webcam. Por meio do modelo **YOLO (You Only Look Once)**, o sistema é capaz de reconhecer cada carta de forma específica, indicando tanto o número ou nome (como "2", "Rei" ou "Ás") quanto o naipe correspondente (como "Copas", "Ouros", "Espadas" ou "Paus"). Essa abordagem permite uma identificação rápida e precisa das cartas, mesmo em condições de variação de iluminação ou ângulo.

## Demonstração
![Demonstração usando 9 de paus e Ás de ouros](https://github.com/FilhoDeEd/Cards-Detector/blob/main/assets/demo.gif)

## Instalação

### Pré-requisitos

- **Python 3**
- **PyTorch**
- **Ultralytics**
- **requirements**
- **Baralho ou imagem de uma carta**

### Passos para Instalar

1. **Instalar o Python no Windows**
   - Baixe o Python no site oficial: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Durante a instalação, marque a opção "Add Python to PATH".
   - Após a instalação, abra o terminal (Prompt de Comando) e verifique a instalação digitando:
     ```bash
     python --version
     ```

2. **Instalar o PyTorch**  
   - A instalação do PyTorch varia dependendo das configurações do seu computador, como o sistema operacional, a versão do Python e se você utiliza CPU ou GPU.  
   - Para garantir uma instalação correta, acesse o site oficial do PyTorch e siga as instruções específicas para o seu ambiente:      [PyTorch - Get Started](https://pytorch.org/get-started/locally/)  

3. **Instalar a Biblioteca Ultralytics**
   - Abra o terminal e execute o seguinte comando para instalar a biblioteca:

      ```bash
      pip install ultralytics
      ```

4. **Instalar as dependências**  
   - Certifique-se de que todas as dependências necessárias sejam instaladas utilizando o arquivo `requirements.txt`.  
   - No terminal, execute o seguinte comando:  
     ```bash
     pip install -r requirements.txt
     ``` 

## Como Usar

1. **Clone o Repositório**

```bash
   git clone https://github.com/FilhoDeEd/Cards-Detector-2.git
```

2. **Execute o código**
- Na pasta do repositório, execute o comando para iniciar a identificação de cartas:
```bash
   python predict.py
```

3. **Aproxime a carta desejada no local indicado**

## Descrição do Código - Criação de Dataset para Detecção de Cartas

1. **Configurações Iniciais**  
   - Define os parâmetros globais, como dimensões das imagens e proporções para dividir o dataset em treino, teste e validação.  
   - Mapeia IDs numéricos às classes das cartas de baralho.

2. **Transformações de Imagens**  
   - Aplica diversas modificações nas imagens das cartas:  
     - **Rotação, Escalonamento e Cisalhamento:** Altera a orientação e o tamanho das cartas.  
     - **Ruído e Brilho:** Introduz variações para simular diferentes condições visuais.  
     - **Borramento:** Suaviza a imagem para testar o desempenho em casos de baixa nitidez.

3. **Sobreposição de Imagens**  
   - Combina imagens transformadas de cartas com fundos variados.  
   - Realiza validação para evitar sobreposição entre as cartas no mesmo fundo.

4. **Anotações e Formato YOLO**  
   - Calcula caixas delimitadoras orientadas para cada carta detectada.  
   - Converte as anotações para o formato YOLO, facilitando o treinamento de modelos de detecção.

5. **Divisão e Organização do Dataset**  
   - Divide automaticamente as imagens geradas em conjuntos de treino, teste e validação.  
   - Salva os arquivos em diretórios separados para imagens e anotações.

6. **Fontes de Dados**  
   - **Imagens de Fundo:** Obtidas de datasets do Kaggle e arquivos ZIP no Google Drive.  
   - **Imagens de Cartas:** Armazenadas localmente no sistema de arquivos.

7. **Execução do Código**  
   - A função `main()` realiza todo o fluxo de geração do dataset, desde o carregamento dos fundos até o armazenamento das imagens e anotações.  
   - Dependências como OpenCV e NumPy são essenciais para o funcionamento.

8. **Personalização**  
   - Ajuste parâmetros de transformação como `ANGLE_RANGE` e `SCALE_RANGE` para alterar as variações nas imagens.  
   - Modifique `TRAIN_SPLIT`, `TEST_SPLIT` e `VALIDATION_SPLIT` para mudar a proporção de divisão do dataset.  

9. **Encerramento**  
   - O pipeline finaliza com a criação de um dataset estruturado, pronto para ser usado em treinamentos de modelos de detecção e reconhecimento de objetos.


## Descrição do Código - Detecção com YOLO

1. **Inicialização do Modelo e Configuração**
   - Carrega o modelo YOLO utilizando o caminho do arquivo `models/best.pt`.
   - Configura uma janela de exibição com o nome "tela".
   - Inicializa a webcam para captura de vídeo. Caso a webcam não esteja acessível, o programa exibe uma mensagem de erro e encerra a execução.

2. **Função de Redimensionamento e Recorte**
   - Define a função `resize_and_crop_to_square`, que redimensiona a imagem mantendo a proporção e corta o centro para obter um frame quadrado de tamanho fixo (960x960 pixels).

3. **Captura e Processamento de Imagens**
   - Lê frames da webcam em um loop contínuo.
   - Redimensiona e recorta cada frame para garantir o formato quadrado necessário para o modelo YOLO.
   - Realiza a inferência no frame processado usando o modelo YOLO, que detecta objetos e retorna resultados anotados.

4. **Exibição de Resultados**
   - Mostra os frames anotados em uma janela chamada "tela".
   - Atualiza a janela em tempo real enquanto o programa processa os frames.

5. **Encerramento**
   - Permite encerrar o programa ao pressionar a tecla 'q', liberando os recursos da webcam e fechando as janelas do OpenCV.


## Autores
- André Lisboa Augusto; 
- Edson Rodrigues da Cruz Filho;
- Marcos Henrique Maimoni Campanella;
- Rodolfo Henrique Raymundo Engelmann;

### Engenharia da Computação - IFSP Piracicaba. Janeiro, 2025