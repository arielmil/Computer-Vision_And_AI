import kagglehub
import os
import numpy as np

path = os.path.expanduser('~/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2')

if not os.path.exists(path):
    print("Downloading dataset...")
    path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
    print("Downloaded dataset to", path)

def load_images_from_folder(folder_path, everything_at_once=False):
    """
    Carrega imagens do dataset em batches para evitar consumo excessivo de memória.
    """
    data, labels = [], []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            num_images = int(len(images))
            for image_file in range(num_images):
                image_file = images[image_file]
                
                data.append(image_file)
                labels.append(int(label))

                if not everything_at_once:
                    # Se a memória estiver alta, libere os dados periodicamente
                    if len(data) % 5000 == 0:
                        print(f"Carregadas {len(data)} imagens, liberando memória...")
                        yield np.array(data), np.array(labels)
                        data, labels = [], []

    if everything_at_once:
        # Retorna todos os dados de uma só vez
        yield np.array(data), np.array(labels)
    else:
        # Retorna o restante dos dados
        yield np.array(data), np.array(labels)

print("Exporting dataset path...")