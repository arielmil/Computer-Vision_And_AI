import torch
import torch.nn as nn
from Genome import Genome

iterations = 0

gen = -1

def eval_genomes(genomes, config, X_train, y_train, debug=False):
    global gen
    
    # Converte os dados para tensores no device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train_torch = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long, device=device)
    
    # Se os rótulos estão one-hot, converte para índices de classe
    if y_train_torch.ndim > 1 and y_train_torch.shape[1] > 1:
        y_train_torch = torch.argmax(y_train_torch, dim=1)
    
    gen += 1
    loss_fn = nn.CrossEntropyLoss()
    
    for genome_id, genome in genomes:
        # Cria a rede PyTorch a partir do genome
        neat_net = Genome(genome, config)
        net = neat_net.decode_genome_to_torch().to(device)
        
        # Forward pass no dataset
        outputs = net(X_train_torch)  # Logits
        
        # Calcula a Cross-Entropy Loss
        loss = loss_fn(outputs, y_train_torch)
        
        # Calcula a fitness como uma função inversa da loss
        fitness = 1 / (1 + loss.item())
        
        if debug:
            print(f"[DEBUG] Gen: {gen} - Genome ID={genome_id} -> Loss={loss.item():.5f}, Fitness={fitness:.5f}")
        
        # Atribui a fitness ao genome
        genome.fitness = fitness
