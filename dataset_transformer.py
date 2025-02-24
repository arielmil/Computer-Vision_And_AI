import cv2
import os
from dataset_manager import path

# Diretórios das imagens
input_folder = path
images_dir = path + "_resized"

if os.path.exists(images_dir):
    print(f"O diretório {images_dir} já existe. Pulando a criação de diretórios e redimensionamento.")
    exit()
    
os.makedirs(images_dir, exist_ok=True)  # Garante que a pasta principal existe

for dir in os.listdir(input_folder):
    category_path = os.path.join(input_folder, dir)
    output_category_path = os.path.join(images_dir, dir)  # Cria a pasta correspondente no output

    # Garante que a pasta de categoria existe no output
    os.makedirs(output_category_path, exist_ok=True)

    files = os.listdir(category_path)

    for filename in files:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(category_path, filename)

            # Ler imagem em escala de cinza
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Verifica se a imagem foi carregada corretamente
            if img is None:
                print(f"Erro ao carregar a imagem: {img_path}")
                continue  # Pula para a próxima imagem

            # Redimensionar para 20x20
            img_resized = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LANCZOS4)

            # Caminho para salvar a imagem redimensionada
            save_path = os.path.join(output_category_path, filename)
            cv2.imwrite(save_path, img_resized)

print("Redimensionamento concluído!")
print("Imagens redimensionadas salvas em", images_dir)