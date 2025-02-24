import cv2
import os
from dataset_manager import path

images_dir = path + "_resized"

def transform_dataset():
    if os.path.exists(images_dir):
        print(f"O diretório {images_dir} já existe. Pulando o redimensionamento.")
        return  # Retorna em vez de encerrar o script

    os.makedirs(images_dir, exist_ok=True)

    for dir in os.listdir(path):
        category_path = os.path.join(path, dir)
        output_category_path = os.path.join(images_dir, dir)
        os.makedirs(output_category_path, exist_ok=True)

        for filename in os.listdir(category_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Erro ao carregar a imagem: {img_path}")
                    continue
                img_resized = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LANCZOS4)
                save_path = os.path.join(output_category_path, filename)
                cv2.imwrite(save_path, img_resized)

    print("Redimensionamento concluído!")
    print("Imagens redimensionadas salvas em", images_dir)

transform_dataset()