import os
import torch
import tensorneat
import pickle
from torchvision import transforms
from PIL import Image

# 🔥 Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {device}")

# 📂 **Função para carregar o dataset**
def load_images_from_folder(folder_path, use_percentage=0.7):
    data, labels = [], []
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            num_images = int(len(images) * use_percentage)
            for image_file in images[:num_images]:
                img_path = os.path.join(label_path, image_file)
                img = Image.open(img_path).convert('L')
                img = transform(img).flatten()
                data.append(img)
                labels.append(int(label))

    return torch.stack(data).to(device), torch.tensor(labels, dtype=torch.long).to(device)

# 📂 **Carregar o dataset**
path = r'C:/Users/mileguir/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2'
X_data, y_data = load_images_from_folder(path, use_percentage=0.7)
print(f"✅ Dataset carregado com {X_data.shape[0]} amostras!")

# 📌 **Definir a Função de Fitness**
def eval_genomes(genomes, config, batch_size=4096):
    """
    Avalia os genomas com base na precisão e confiança.
    """
    num_samples = X_data.shape[0]
    alpha, beta = 0.7, 0.3  # Pesos para precisão e confiança

    for genome_id, genome in genomes:
        model = tensorneat.NEATModel(genome, config).to(device)
        model.eval()

        total_correct, total_confidence = 0, 0

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_X = X_data[i:i + batch_size]
                batch_y = y_data[i:i + batch_size]

                output = torch.softmax(model(batch_X), dim=1)
                predicted_labels = torch.argmax(output, dim=1)
                correct_probs = output[range(len(batch_y)), batch_y]

                total_correct += (predicted_labels == batch_y).sum().item()
                total_confidence += correct_probs.sum().item()

        accuracy = total_correct / num_samples
        confidence = total_confidence / num_samples
        fitness = alpha * accuracy + beta * confidence

        genome.fitness = fitness
        print(f"🧬 Genoma {genome_id} | Fitness: {fitness:.4f} | Accuracy: {accuracy:.4f} | Confidence: {confidence:.4f}")

# 🚀 **Configurar e Executar o TensorNEAT**
def run_neat():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")

    config = tensorneat.NEATConfig(config_path)
    trainer = tensorneat.Trainer(config)

    print("🚀 Iniciando evolução com TensorNEAT...")
    winner = trainer.run(eval_genomes, generations=50)
    print("🏆 Evolução concluída!")

    # Salvar o modelo vencedor
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("✅ Melhor genoma salvo como 'best_genome.pkl'!")

    return winner

# Executar o NEAT
if __name__ == "__main__":
    best_model = run_neat()