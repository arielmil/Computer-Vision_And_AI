from NEAT import *
from dataset_transformer import images_dir
from dataset_manager import load_images_from_folder

from sklearn.model_selection import train_test_split

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome, DefaultMutation, DefaultCrossover, DefaultDistance, DefaultConn
from tensorneat.common import ACT, AGG
from tensorneat.genome.gene.node.bias import BiasNode

import pickle

# Carregar o dataset
X_data, y_data = [], []
for X_part, y_part in load_images_from_folder(images_dir, everything_at_once=False):
    X_data.append(X_part)
    y_data.append(y_part)

# Concatena os lotes de forma eficiente
X_data = np.concatenate(X_data, axis=0)
y_data = np.concatenate(y_data, axis=0)

# Converter labels para one-hot encoding
y_data = np.eye(10)[y_data]

# Dividir dataset em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# Criar problema de aprendizado supervisionado
supervised_problem = CustomSupervisedFuncFit(X_train, y_train, batch_size=32)

# Configura a arquitetura da rede neural
genome = DefaultGenome(
    num_inputs=20*20,  # Número de pixels da imagem (entrada)
    num_outputs=10,  # 10 classes (0-9)
    max_nodes = 30000, # Número máximo de neurônios
    max_conns = 250000, # Número máximo de conexões
    mutation = DefaultMutation(), # Mutação padrão
    crossover = DefaultCrossover(), # Crossover padrão
    distance = DefaultDistance(), # Distância padrão
    init_hidden_layers=(),  # Deixa o NEAT evoluir a estrutura oculta
    node_gene=BiasNode(
        activation_options=[ACT.sigmoid],  # Ativação Sigmoid nos neurônios ocultos
        aggregation_options=[AGG.sum, AGG.product],  # Opções de agregação
    ),
    conn_gene = DefaultConn(), # Conexão padrão
    output_transform=softmax,  # Softmax na saída para classificação multiclasse
)

# Configura o algoritmo NEAT
algorithm = NEAT(
    pop_size=200,  # Tamanho da população
    species_size=20,  # Número de espécies na população
    survival_threshold=0.01,  # Percentual de sobrevivência por geração
    genome=genome,  # Usa o genoma configurado
)

# Configurar e rodar o pipeline NEAT
pipeline = Pipeline(
    algorithm=algorithm,
    problem=supervised_problem,
    generation_limit=50,
    fitness_target=-0.01,
    seed=42,
)

# Inicializar o estado do NEAT
state = pipeline.setup()

# Executar evolução
state, best = pipeline.auto_run(state)

# Testar a melhor rede neural no conjunto de teste
best_network = best.make_network()
test_predictions = np.array([best_network(x) for x in X_test])

# Calcular acurácia
test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
print(f"🎯 Acurácia final no conjunto de teste: {test_accuracy * 100:.2f}%")

# Salvar modelo treinado
with open("best_neat_model.pkl", "wb") as f:
    pickle.dump(best, f)
print("✅ Modelo salvo como 'best_neat_model.pkl'")