import neat
import numpy as np
import pickle

# Carregar dataset simulado
def load_dataset(size, pixels):
    X = np.random.rand(size, pixels**2)  # size imagens simuladas, cada xi em X é um vetor de pixels^2 pixels.
    y = np.random.randint(0, 10, size)  # Labels aleatórios de 0 a 9
    return X, y

# Função de avaliação da população
def eval_genomes(genomes, config):
    X, y = load_dataset()
    num_samples = len(X)

    generation_errors = []  # Lista para armazenar os erros individuais dos genomas

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_error = 0

        for i in range(num_samples):
            output = net.activate(X[i])  # Calcula a saída da rede
            correct_label = y[i]  # Obtém o valor real
            predicted_prob = output[correct_label]  # Probabilidade da classe correta
            
            # Cálculo do MSE específico
            error = (correct_label - (predicted_prob * correct_label)) ** 2
            total_error += error

        mse = total_error / num_samples  # Calcula o MSE médio
        genome.fitness = 1 / (mse + 1e-6)  # Inverte o erro para maximizar o fitness

        generation_errors.append(mse)  # Salva o erro do genoma

    # Estatísticas da geração
    best_genome = max(genomes, key=lambda g: g[1].fitness)  # Melhor indivíduo da geração
    best_error = min(generation_errors)  # Melhor erro (menor MSE)
    mean_error = np.mean(generation_errors)  # Erro médio da geração

    print(f"Geração {pop.generation}: Melhor Fitness = {best_genome[1].fitness:.6f}, "
          f"Melhor Erro (MSE) = {best_error:.6f}, Erro Médio = {mean_error:.6f}")

# Função principal para rodar o NEAT
def run_neat():
    config_path = "config-feedforward.txt"
    config = neat.Config(neat.DefaultGenome, 
                         neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, 
                         neat.DefaultStagnation, 
                         config_path)
    
    global pop  # Para acessar na função de avaliação
    pop = neat.Population(config)

    # Relatórios para acompanhar a evolução
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Executa a evolução por 50 gerações
    winner = pop.run(eval_genomes, 50)

    # Salva o melhor modelo treinado
    with open("best_neat_model.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner

# Rodar NEAT
best_model = run_neat()