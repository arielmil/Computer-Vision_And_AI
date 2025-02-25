import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import torch
import neat
from network_renderizer import draw_neural_net

# Suppose these are local modules in your project
from dataset_transformer import images_dir
from dataset_manager import load_images_from_folder

# Import the custom NEAT code
from NEAT import eval_genomes, Genome

from network_renderizer import draw_neural_net



def main():
    # 1) Load data
    X_data, y_data = [], []
    for X_part, y_part in load_images_from_folder(images_dir, everything_at_once=False):
        X_data.append(X_part)
        y_data.append(y_part)

    # Concat all partial data
    X_data = np.concatenate(X_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)
    
    # If your dataset_manager gives integer labels 0..9, 
    # but your code snippet shows:
    #       y_data = np.eye(10)[y_data]
    # That’s one-hot. Typically for cross-entropy, we don’t want one-hot in training. 
    # However, if you *did* create one-hot above, let's keep it for now 
    # (and in eval_genomes, we call .argmax). 
    #
    # If you never turned them into one-hot, no problem, just keep them as integer class labels.

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, 
        y_data, 
        test_size=0.3, 
        random_state=42
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    # 3) Set up NEAT
    #    Make sure you have a config file (like 'config-feedforward') 
    #    properly pointing to your neat-python parameters.

    current_dir = os.path.basename(os.getcwd())
    if current_dir != "NEAT-Algorithms":
        os.chdir("NEAT-Algorithms")
    config_path = "config-feedforward.ini"

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create population
    pop = neat.Population(config)

    # Add reporters so we see progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # 4) Run NEAT. We pass a function that takes (genomes, config),
    #    but we also want X_train, y_train inside it. 
    #    A quick trick is to use a lambda or partial:
    n_generations = 500
    def eval_genomes_with_data(genomes, config):
        eval_genomes(genomes, config, X_train, y_train)

    winner = pop.run(eval_genomes_with_data, n_generations)
        
    # Optionally, evaluate winner on test set
    #-----------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_genome_torch = Genome(winner, config).decode_genome_to_torch().to(device)

    # Convert test to torch
    X_test_torch = torch.tensor(X_test, dtype=torch.float, device=device)
    # If your test labels are one-hot, make them class indices:
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_indices = y_test.argmax(axis=1)
    else:
        y_test_indices = y_test
    y_test_torch = torch.tensor(y_test_indices, dtype=torch.long, device=device)

    # Forward pass
    outputs = best_genome_torch(X_test_torch)  # shape [N, 10], presumably
    preds = outputs.argmax(dim=1).detach().cpu().numpy()
    accuracy = np.mean(preds == y_test_indices)
    print(f"Accuracy on test set: {accuracy*100:.2f}%")

    # 5) Save the best genome
    with open("best_neat_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Saved best genome to 'best_neat_genome.pkl'")

    # 6) Draw the neural network
    draw_neural_net(winner, config, "winner_topology")

if __name__ == "__main__":
    main()