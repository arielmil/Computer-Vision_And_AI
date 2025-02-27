import neat
import numpy as np
import pickle
import os
import graphviz
import neat

def draw_neural_net(genome, config, filename="network"):
    """
    Desenha a topologia da rede neural evoluída pelo NEAT organizando corretamente os nós em camadas.

    Argumentos:
    - genome: O melhor genoma (winner).
    - config: Configuração do NEAT.
    - filename: Nome do arquivo de saída.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR")  # Esquerda para direita (horizontal)

    # Obter nós de entrada e saída do config
    input_nodes = list(range(-config.genome_config.num_inputs, 0))
    output_nodes = list(range(config.genome_config.num_outputs))

    # Identificar nós ocultos dinamicamente
    hidden_nodes = [n for n in genome.nodes.keys() if n not in input_nodes and n not in output_nodes]

    # Criar subgrafos para organizar as camadas
    with dot.subgraph() as s:
        s.attr(rank="same")  # Mesma posição vertical
        for node in input_nodes:
            s.node(str(node), f"Input {node}", shape="circle", style="filled", fillcolor="lightgray")

    with dot.subgraph() as s:
        s.attr(rank="same")
        for node in hidden_nodes:
            s.node(str(node), f"Hidden {node}", shape="circle", style="filled", fillcolor="gray")

    with dot.subgraph() as s:
        s.attr(rank="same")
        for node in output_nodes:
            s.node(str(node), f"Output {node}", shape="circle", style="filled", fillcolor="red")

    # Criar as conexões
    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        src, dst = conn.key
        if src in output_nodes and dst in output_nodes:
            continue  # Evita conexões diretas entre nós de saída
        dot.edge(str(src), str(dst), label=f"{conn.weight:.2f}")

    # Gerar e visualizar a rede neural
    dot.render(filename, view=True)

    print(f"Topologia salva como {filename}.png")

# XOR dataset
data = [
    (np.array([0.0, 0.0]), np.array([0.0])),
    (np.array([0.0, 1.0]), np.array([1.0])),
    (np.array([1.0, 0.0]), np.array([1.0])),
    (np.array([1.0, 1.0]), np.array([0.0]))
]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 4.0  # Começa com fitness máximo possível
        
        for inputs, expected in data:
            output = net.activate(inputs)[0]
            fitness -= (output - expected[0]) ** 2  # Erro quadrático
        
        genome.fitness = fitness
        print(f"Genome {genome_id} - Size: {genome.size()} - Fitness: {fitness:.5f}")

# Carregar config padrão do NEAT
os.chdir("tests")
config_path = "config.ini"


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Criar população e reportar os tamanhos das redes
pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# Rodar a evolução por algumas gerações
winner = pop.run(eval_genomes, 10)

# Imprimir tamanho final do melhor genoma
tamanho_final = winner.size()
print(f"\nMelhor genoma final: Size = {tamanho_final}")

with open("best_neat_genome_test.pkl", "wb") as f:
        pickle.dump(winner, f)
print("Saved best genome to 'best_neat_genom_test.pkl'")

# Gerar e visualizar a rede neural do winner
draw_neural_net(winner, config, "winner_topology")

print("Nós do Winner:", list(winner.nodes.keys()))
