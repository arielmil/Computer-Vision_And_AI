import graphviz

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