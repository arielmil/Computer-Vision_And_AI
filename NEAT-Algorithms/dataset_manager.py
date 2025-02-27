import kagglehub
import os
import numpy as np
import cv2

path = os.path.expanduser('~/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2')

if not os.path.exists(path):
    print("Downloading dataset...")
    path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
    print("Downloaded dataset to", path)

def load_images_from_folder(folder_path, everything_at_once=False, flatten=True):
    """
    Carrega imagens do dataset em batches para evitar consumo excessivo de memória.
    """
    data, labels = [], []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            num_images = int(len(images))
            for idx in range(num_images):
                image_file = images[idx]
                img_path = os.path.join(label_path, image_file)

                # Carrega a imagem como escala de cinza
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Normaliza e converte o array para float32
                img = (img.astype(np.float32) / 255.0)

                if flatten:
                    img = img.flatten()

                data.append(img)
                labels.append(int(label))

                if not everything_at_once:
                    # Se a memória estiver alta, libere os dados periodicamente
                    if len(data) % 5000 == 0:
                        print(f"Carregadas {len(data)} imagens, liberando memória...")
                        yield np.array(data), np.array(labels)
                        data, labels = [], []

print("Exporting dataset path...")