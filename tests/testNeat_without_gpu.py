import neat
import time
import numpy as np
from neat.nn import FeedForwardNetwork
from sklearn.model_selection import train_test_split
import pickle

import sys
import os

# Adiciona o diretório pai ao sys.path para permitir importação dos módulos superiores
NEAT_ALGORITHMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(NEAT_ALGORITHMS_DIR)

from dataset_transformer import images_dir
from dataset_manager import load_images_from_folder
from NEAT import eval_genomes as eval_genomes_with_gpu, Genome
from network_renderizer import draw_neural_net


# Configuração inicial
gen = -1

def eval_genomes_with_cpu(genomes, config, X_train, y_train, debug=False):
    global gen
    gen += 1
    
    # Se os rótulos estão one-hot, converte para índices de classe
    y_train = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
    
    loss_fn = lambda outputs, targets: np.mean(-np.log(outputs[np.arange(len(targets)), targets] + 1e-9))
    
    for genome_id, genome in genomes:
        # Cria a rede diretamente no neat-python
        net = FeedForwardNetwork.create(genome, config)
        
        # Forward pass no dataset
        outputs = np.array([net.activate(x) for x in X_train])
        
        # Normaliza os outputs para probabilidades usando softmax
        exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        outputs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        
        # Calcula a Cross-Entropy Loss
        loss = loss_fn(outputs, y_train)
        
        # Fitness é inversamente proporcional à perda
        fitness = 1 / (1 + loss)
        
        if debug:
            print(f"[DEBUG] Gen: {gen} - Genome ID={genome_id} -> Loss={loss:.5f}, Fitness={fitness:.5f}")
        
        # Atribui a fitness ao genome
        genome.fitness = fitness


# Função principal para rodar o NEAT
def run_neat(config_path, X_train, y_train, generations=50, debug=False, with_gpu = False):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Criar população
    population = neat.Population(config)
    
    # Adicionar repórter para visualização
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    if with_gpu:
        # Rodar o algoritmo NEAT com GPU
        winner = population.run((lambda genomes, config: eval_genomes_with_gpu(genomes, config, X_train, y_train, debug)), generations)

    else:
        # Rodar o algoritmo NEAT tradicional
        winner = population.run((lambda genomes, config: eval_genomes_with_cpu(genomes, config, X_train, y_train, debug)), generations)

    return winner

def main():
    # Para usar no draw_neural_net
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    X_data, y_data = [], []
    for X_part, y_part in load_images_from_folder(images_dir, everything_at_once=False):
        X_data.append(X_part)
        y_data.append(y_part)
    
    # Concat all partial data
    X_data = np.concatenate(X_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, 
        y_data, 
        test_size=0.3, 
        random_state=42
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    current_dir = os.path.basename(os.getcwd())
    if current_dir != "NEAT-Algorithms":
        os.chdir(NEAT_ALGORITHMS_DIR)
    config_path = "config-feedforward.ini"

    start_time = time.time()
    
    winner = run_neat(config_path, X_train, y_train, generations=5, debug=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    print(f"Tempo total: (Sem GPU) {int(hours):02}:{int(minutes):02}:{int(seconds):02}:{int(milliseconds):03}")

    with open("best_neat_genome_test_cpu.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Saved best genome to 'best_neat_genome_test_cpu.pkl'")

    draw_neural_net(winner, config , "winner_topology")

    start_time = time.time()
    
    winner = run_neat(config_path, X_train, y_train, generations=5, debug=False, with_gpu = True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    print(f"Tempo total: (Sem GPU) {int(hours):02}:{int(minutes):02}:{int(seconds):02}:{int(milliseconds):03}")

    with open("best_neat_genome_test_gpu.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Saved best genome to 'best_neat_genome_test_gpu.pkl'")

    draw_neural_net(winner, config , "winner_topology")

if __name__ == "__main__":
    main()

