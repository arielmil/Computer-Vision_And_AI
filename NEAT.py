import os
import tensorneat as tn
import pickle
from torchvision import transforms
from PIL import Image
import kagglehub
import configparser

def convert_to_utf8(file_path):
    try:
        with open(file_path, "rb") as f:
            raw_content = f.read()

        for encoding in ["utf-8", "utf-16", "latin-1", "windows-1252"]:
            try:
                decoded_content = raw_content.decode(encoding)
                print(f"✅ Arquivo config.ini lido corretamente usando {encoding}")

                cleaned_content = decoded_content.encode("utf-8", "ignore").decode("utf-8")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)

                print("✅ Arquivo config.ini convertido para UTF-8 e caracteres inválidos removidos!")
                return

            except UnicodeDecodeError:
                continue

        print("❌ Não foi possível converter config.ini para UTF-8. Verifique o arquivo manualmente.")

    except Exception as e:
        print(f"❌ Erro inesperado ao processar config.ini: {e}")


def eval_genomes(genomes, config, batch_size=4096):
    num_samples = len(X_data)
    alpha, beta = 0.7, 0.3

    for genome_id, genome in genomes:
        model = tn.NEATModel(genome, config)
        model.eval()

        total_correct, total_confidence = 0, 0

        for i in range(0, num_samples, batch_size):
            batch_X = X_data[i:i + batch_size]
            batch_y = y_data[i:i + batch_size]

            output = model.activate(batch_X)
            predicted_labels = [max(range(len(o)), key=lambda k: o[k]) for o in output]
            correct_probs = [o[y] for o, y in zip(output, batch_y)]

            total_correct += sum(p == y for p, y in zip(predicted_labels, batch_y))
            total_confidence += sum(correct_probs)

        accuracy = total_correct / num_samples
        confidence = total_confidence / num_samples
        fitness = alpha * accuracy + beta * confidence

        genome.fitness = fitness
        print(f"🧬 Genoma {genome_id} | Fitness: {fitness:.4f} | Accuracy: {accuracy:.4f} | Confidence: {confidence:.4f}")


def run_neat():
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Erro: O arquivo de configuração '{config_path}' não foi encontrado.")

        convert_to_utf8(config_path)

        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            eval_instance_count = config.getint('TENSORNEAT', 'eval_instance_count', fallback=1)
        except Exception as e:
            print(f"❌ Erro ao carregar a configuração do TensorNEAT: {e}")
            return None

        try:
            backup_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup")
            os.makedirs(backup_dir_path, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível criar o diretório de backup: {e}")

        try:
            trainer = tn.NEATEngine(config, eval_genomes, backup_dir_path, eval_instance_count=eval_instance_count)
        except Exception as e:
            print(f"❌ Erro ao criar a engine de evolução: {e}")
            return None

        print("🚀 Iniciando evolução com TensorNEAT...")
        try:
            winner = trainer.run(generations=50)
        except Exception as e:
            print(f"❌ Erro durante a execução da evolução: {e}")
            return None

        print("🏆 Evolução concluída!")

        try:
            with open("best_genome.pkl", "wb") as f:
                pickle.dump(winner, f)
            print("✅ Melhor genoma salvo como 'best_genome.pkl'!")
        except Exception as e:
            print(f"⚠️ Erro ao salvar o melhor genoma: {e}")

        return winner

    except Exception as e:
        print(f"❌ Erro inesperado na função run_neat: {e}")
        return None

if __name__ == "__main__":
    best_model = run_neat()


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

    return data, labels

path = r'C:/Users/mileguir/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2'
if not os.path.exists(path):
    print(f"Dataset não encontrado em {path}. Baixando...")
    path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
else:
    print(f"Dataset encontrado em {path}.")

X_data, y_data = load_images_from_folder(path, use_percentage=0.7)
print(f"✅ Dataset carregado com {len(X_data)} amostras!")