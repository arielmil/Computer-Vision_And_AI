import graphviz
import pickle
import neat
import os

def draw_net(config, genome, filename="neural_net", view=False, node_names=None):
    """Desenha a topologia da rede neural usando Graphviz."""
    if node_names is None:
        node_names = {}

    dot = graphviz.Digraph(format="png", engine="dot")
    dot.attr(rankdir="LR")

    # Adicionar um único nó representando todos os inputs
    dot.node("Input Layer", shape="box", style="filled", color="lightblue")

    # Adiciona os nós de saída
    outputs = config.genome_config.output_keys
    for o in outputs:
        name = node_names.get(o, f"Output {o}")
        dot.node(name, shape="circle", style="filled", color="lightgreen")

    # Adiciona os nós ocultos
    for n in genome.nodes:
        if n not in config.genome_config.input_keys and n not in outputs:
            name = node_names.get(n, str(n))
            dot.node(name, shape="circle", style="filled", color="lightgrey")

    # Adiciona as conexões
    for key, cg in genome.connections.items():
        if cg.enabled:
            input_idx, output_idx = key
            input_name = "Input Layer" if input_idx in config.genome_config.input_keys else node_names.get(input_idx, str(input_idx))
            dot.edge(input_name, node_names.get(output_idx, str(output_idx)), label=f"{cg.weight:.2f}")

    # Renderiza e salva
    dot.render(filename, view=view)
    print(f"Rede neural salva como {filename}.png")

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

# Carregar e visualizar a melhor rede treinada
with open("best_neat_model.pkl", "rb") as f:
    best_genome = pickle.load(f)

# Nomear nós de entrada e saída
node_names = {i: f"Input {i}" for i in range(config.genome_config.num_inputs)}
node_names.update({config.genome_config.num_inputs + i: f"Output {i}" for i in range(config.genome_config.num_outputs)})

draw_net(config, best_genome, "best_neat_network", node_names=node_names)
