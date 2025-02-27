from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import os

# Forçar uso do backend sem X11 para evitar erro "BadValue (X_CreatePixmap)"
import matplotlib
matplotlib.use('Agg')  # Usa o modo de salvamento sem tentar renderizar na tela

# Função para extrair o número de neurônios por camada
def get_neural_network_structure(model):
    layer_sizes = []
    for layer in model.layers:
        try:
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if isinstance(shape, list):
                    shape = shape[0]

                # Se for Flatten, captura corretamente
                if isinstance(layer, tf.keras.layers.Flatten):
                    neurons = shape[-1]  
                elif isinstance(layer, tf.keras.layers.Dense):
                    neurons = shape[-1]
                elif isinstance(layer, tf.keras.layers.Conv2D):
                    neurons = shape[1] * shape[2] * shape[3]
                else:
                    neurons = shape[-1]  

                # Evita adicionar "1" como número de neurônios inválido
                if neurons > 1:
                    layer_sizes.append(neurons)

                print(f"Camada: {layer.name} | Output Shape: {shape} | Neurônios: {neurons}")
        except Exception as e:
            print(f"Camada ignorada: {layer.name} | Erro: {e}")

    print("Estrutura da rede:", layer_sizes)
    return layer_sizes

# Função para plotar a topologia da rede neural
def plot_neural_network(layers, filename="arquitetura_rede.png"):
    if not layers:
        print("Erro: Nenhuma camada encontrada para plotar.")
        return

    G = nx.DiGraph()
    positions = {}

    y_offset = 0
    for layer_idx, num_neurons in enumerate(layers):
        for neuron in range(num_neurons):
            node_id = f"L{layer_idx}_N{neuron}"
            G.add_node(node_id, layer=layer_idx)
            positions[node_id] = (layer_idx * 2, -neuron + y_offset)

        y_offset += num_neurons / 2  

    for layer_idx in range(len(layers) - 1):
        for src in range(layers[layer_idx]):
            for dst in range(layers[layer_idx + 1]):
                src_id = f"L{layer_idx}_N{src}"
                dst_id = f"L{layer_idx+1}_N{dst}"
                G.add_edge(src_id, dst_id)

    width = max(10, len(layers) * 3)
    height = max(5, max(layers) // 2)

    plt.figure(figsize=(width, height))
    nx.draw(G, pos=positions, with_labels=False, node_size=300, node_color="lightblue", edge_color="black")
    plt.title("Arquitetura da Rede Neural")

    # Salva a imagem em vez de exibir na tela
    plt.savefig(filename, dpi=300)
    print(f"Imagem da arquitetura da rede salva em {filename}")

curPath = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.abspath(os.path.join(curPath, "autokeras", "melhor_modelo_autokeras.keras"))

# Carrega o modelo salvo
modelo_carregado = load_model(modelo_path)

# Exibe a estrutura do modelo
modelo_carregado.summary()

# Obter a estrutura da rede
layer_sizes = get_neural_network_structure(modelo_carregado)

# Salva um diagrama da rede no formato SVG
plot_model(
    modelo_carregado, 
    to_file="neural-networks/svg-files/arquitetura_rede.svg",
    show_shapes=True, 
    show_layer_names=True,
    expand_nested=True,  
    rankdir="LR",  
    dpi=300  
)

# Plotar o gráfico da rede neural e salvar
# plot_neural_network(layer_sizes, "arquitetura_rede.svg")
