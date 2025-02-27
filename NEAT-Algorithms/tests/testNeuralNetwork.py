import pickle
import neat
import torch
import numpy as np
import os
import cv2

# Função para carregar os últimos 30% das imagens de cada pasta
def load_images_from_folder(folder_path, use_percentage=0.3):
    data = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            num_images = int(len(images) * use_percentage)  # Quantidade de imagens (30%)
            start_index = len(images) - num_images  # Pega os 30% finais
            for image_file in images[start_index:]:
                image_path = os.path.join(label_path, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = img / 255.0  # Normaliza para [0, 1]
                    img_flatten = img.flatten()
                    data.append(img_flatten)
                    labels.append(int(label))
    return np.array(data), np.array(labels)

# Recriar o objeto de configuração do NEAT
local_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(local_dir, "config.ini")

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

# Carregar o melhor genoma treinado
with open("best_neat_model.pkl", "rb") as f:
    best_genome = pickle.load(f)

# Carregar os últimos 30% do dataset de teste
test_path = r"C:/Users/mileguir/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2"
X_test, y_test = load_images_from_folder(test_path, use_percentage=0.7)

print(f"Total de imagens de teste carregadas (30% finais): {len(X_test)}")

# Criar a função para ativar a rede neural a partir do genoma
def activate_network(genome, config, inputs):
    # Inicializar valores de saída para todos os nós
    outputs = {node: 0.0 for node in genome.nodes}

    # Configurar os valores dos nós de entrada
    for i, input_value in enumerate(inputs):
        outputs[config.genome_config.input_keys[i]] = input_value

    # Processar as conexões
    for (input_node, output_node), conn in genome.connections.items():
        if conn.enabled:
            if input_node in outputs:
                outputs[output_node] += outputs[input_node] * conn.weight

    # Coletar as saídas
    output_values = [outputs[output_node] for output_node in config.genome_config.output_keys]
    return torch.sigmoid(torch.tensor(output_values))  # Função de ativação sigmoidal nas saídas

# Avaliar o genoma no dataset de teste
correct = 0
total = len(X_test)

for i in range(total):
    input_data = X_test[i]
    output = activate_network(best_genome, config, input_data)  # Predições do modelo
    predicted_label = torch.argmax(output).item()
    if predicted_label == y_test[i]:
        correct += 1

accuracy = correct / total * 100
print(f"Precisão no dataset de teste (30% finais): {accuracy:.2f}%")