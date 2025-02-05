import graphviz
import neat.visualize

# Função para desenhar a topologia da rede
def draw_net(pixels, config, genome, filename="neural_net"):
    node_names = {i: f"Pixel {i}" for i in range(pixels**2)}  # Nomeando os neurônios de entrada 0 ... (pixels^2) - 1
    for i in range(10):  # Nomeando os neurônios de saída
        node_names[pixels**2 + i] = f"Output {i}"
    
    # Criamos um gráfico da topologia da rede neural
    graph = neat.visualize.draw_net(config, genome, True, node_names=node_names)
    graph.render(filename, format="png", cleanup=True)  # Salva a imagem da rede neural

# Carregar e visualizar a melhor rede treinada
with open("best_neat_model.pkl", "rb") as f:
    best_genome = pickle.load(f)

draw_net(pixels, config, best_genome, "best_neat_network")