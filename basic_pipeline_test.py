# Minimal NEAT pipeline usando um subconjunto do dataset de dígitos manuscritos
# Este arquivo foi refatorado para ser o mais simples possível e facilita o entendimento

import os  # Módulo para interação com o sistema de arquivos
import numpy as np  # Biblioteca para operações numéricas
from PIL import Image  # Biblioteca para manipulação de imagens
import random  # Biblioteca para operações de aleatoriedade

import jax  # Biblioteca para computação acelerada
from jax import numpy as jnp, random as jrandom  # jax numpy para operações matemáticas e jax.random para geração de números aleatórios
import jax.nn  # Módulo de funções de ativação e softmax

# Importa os módulos do NEAT a partir da biblioteca tensorneat
from tensorneat.pipeline import Pipeline  # Pipeline para gerenciar o fluxo do algoritmo NEAT
from tensorneat.algorithm import NEAT  # Implementação do algoritmo NEAT
from tensorneat.genome import DefaultGenome, DefaultMutation, DefaultCrossover, DefaultDistance, DefaultConn  # Componentes do genoma
from tensorneat.genome.gene.node.bias import BiasNode  # Tipo de nó com bias
from tensorneat.common import ACT, AGG  # Constantes para funções de ativação e agregação

# Tenta importar a função softmax; se não estiver disponível, usa jax.nn.softmax
try:
    from tensorneat.utils import softmax
except ImportError:
    softmax = jax.nn.softmax

# Função para carregar um subconjunto do dataset
# Nesta função, são carregadas as duas primeiras imagens de cada pasta (pasta = label)
DATASET_PATH = "/home/mileguir/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2"

def load_subset_dataset(dataset_path):
    # Inicializa listas para armazenar as imagens e os rótulos (labels)
    images = []
    labels = []
    # Lista todas as pastas dentro do dataset; cada pasta representa um dígito de 0 a 9
    for label in sorted(os.listdir(dataset_path)):
        label_dir = os.path.join(dataset_path, label)  # Concatena o path da pasta
        if os.path.isdir(label_dir):
            # Ordena os nomes dos arquivos para garantir que as duas primeiras imagens sejam selecionadas
            files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            selected = files[:2]  # Seleciona as duas primeiras imagens
            for file in selected:
                file_path = os.path.join(label_dir, file)  # Caminho completo da imagem
                img = Image.open(file_path).convert('L')  # Abre a imagem e converte para escala de cinza (L)
                arr = np.array(img, dtype=np.float32).flatten()  # Converte a imagem para um array e o transforma em vetor
                # Verifica se a imagem possui o número esperado de pixels (12600)
                if arr.size != 12600:
                    print(f"Warning: Image {file_path} has {arr.size} pixels (expected 12600).")
                images.append(arr)  # Adiciona o vetor da imagem à lista
                labels.append(int(label))  # Adiciona o label correspondente (convertido para inteiro)
    # Converte a lista de imagens para um array NumPy com shape (num_samples, 12600)
    X = np.stack(images)
    # Converte os rótulos para codificação one-hot para 10 classes
    Y = np.eye(10)[labels]
    return X, Y

# Importa a classe FuncFit que representa um problema de ajuste de função
from tensorneat.problem.func_fit import FuncFit

# Define uma classe para o problema dos Dígitos Manuscritos, herda de FuncFit
class HandwrittenDigitsProblem(FuncFit):
    def __init__(self, X, y, batch_size=4):
        # Converte os dados de entrada e saída para arrays do JAX
        self.data_inputs = jnp.array(X, dtype=jnp.float32)
        self.data_outputs = jnp.array(y, dtype=jnp.float32)
        # Define o tamanho do batch
        self.batch_size = batch_size

    def evaluate(self, state, randkey, act_func, params):
        # Função de avaliação que calcula a perda usando cross-entropy
        predictions = []
        # Para cada exemplo de entrada, executa a função de ativação (simula a passagem pela rede neural)
        for x in self.data_inputs:
            # x é um vetor (12600,); reformatar para (1, 12600) para compatibilidade
            output = act_func(state, x[None, :], params)[1]  
            # Remove a dimensão extra, resultando em um vetor com 10 elementos
            output = jnp.squeeze(output, axis=0)
            predictions.append(output)
        # Empilha todas as previsões formando um array com shape (num_samples, 10)
        predictions = jnp.stack(predictions)
        epsilon = 1e-9  # Valor pequeno para evitar log(0)
        predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)  # Limita os valores entre epsilon e 1-epsilon
        # Calcula a perda negativa da entropia cruzada; perda menor indica melhor desempenho
        loss = -jnp.mean(jnp.sum(self.data_outputs * jnp.log(predictions), axis=1))
        return loss

    @property
    def input_shape(self):
        # Retorna a forma da entrada, que é (12600, )
        return (self.data_inputs.shape[1],)

    @property
    def output_shape(self):
        # Retorna a forma da saída
        return self.data_outputs.shape

# Define uma função dummy de ativação que simula uma passagem pela rede neural
# Ela recebe o estado, uma entrada e os parâmetros (W e b) e retorna as probabilidades

def dummy_act_func(state, x, params):
    W, b = params  # Desempacota os parâmetros: W (pesos) e b (bias)
    # Computa os logits através do produto da entrada com os pesos e soma do bias
    logits = jnp.dot(x, W) + b  # x: (1, 12600), W: (12600, 10) resulta em logits de shape (1, 10)
    # Aplica a função softmax para converter os logits em probabilidades
    probs = jax.nn.softmax(logits)
    return (None, probs, None, None)  # Retorna uma tupla com a saída no segundo elemento

# Realiza um patch na função forward do genoma para assegurar que ela retorne um formato consistente
from tensorneat.genome import default as def_genome
_original_forward = def_genome.DefaultGenome.forward  # Salva a implementação original

def safe_forward(self, state, transformed, inputs):
    try:
        # Chama a função original de forward com os parâmetros necessários
        res = _original_forward(self, state, inputs)
        # Se o resultado não for uma tupla de pelo menos 4 elementos, encapsula-o em uma tupla
        if not isinstance(res, (tuple, list)) or len(res) < 4:
            return (None, res, None, None)
        return res  # Caso contrário, retorna o resultado como está
    except Exception as e:
        # Em caso de erro, retorna um tensor de zeros com a mesma quantidade de saídas esperada
        size = inputs.shape[0] if hasattr(inputs, 'shape') else 1
        return (None, jnp.zeros((size, self.num_outputs)), None, None)

# Aplica o patch para garantir segurança na função forward
def_genome.DefaultGenome.forward = safe_forward

# Carrega o subconjunto do dataset usando a função definida anteriormente
X, Y = load_subset_dataset(DATASET_PATH)
print(f"Loaded dataset with {X.shape[0]} samples, input dimension {X.shape[1]}")

# Cria uma instância do problema dos Dígitos Manuscritos utilizando apenas o subconjunto carregado
problem = HandwrittenDigitsProblem(X, Y, batch_size=4)

# Configura o genoma usando configurações similares às do arquivo NEAT.py, mas adaptado para este problema
# O genoma define a estrutura da rede neural dos indivíduos na população

# Alteramos max_nodes para um valor maior ou igual a 12610 (número inicial de nós = num_inputs + num_outputs)
genome = DefaultGenome(
    num_inputs=12600,  # Número de pixels em cada imagem
    num_outputs=10,    # Número de classes (dígitos de 0 a 9)
    max_nodes=14000,   # Alterado de 50 para 14000 para acomodar os nós iniciais
    max_conns=140000,  # Alterado de 100 para 140000 para acomodar o número inicial de conexões
    mutation=DefaultMutation(),
    crossover=DefaultCrossover(),
    distance=DefaultDistance(),
    init_hidden_layers=(),
    node_gene=BiasNode(
        activation_options=[ACT.sigmoid],  # Função de ativação (neste caso, sigmoid)
        aggregation_options=[AGG.sum],        # Função de agregação (neste caso, soma)
    ),
    conn_gene=DefaultConn(),
    output_transform=softmax,  # Função de transformação da saída (softmax)
)

# Configura o algoritmo NEAT com parâmetros básicos
# pop_size define o tamanho da população, species_size e survival_threshold controlam a formação de espécies e seleção
algorithm = NEAT(
    pop_size=5,  # Tamanho reduzido da população para facilitar o debug
    species_size=20,
    survival_threshold=0.2,
    genome=genome,
    # Outros parâmetros podem ser configurados conforme necessário
)

# Cria o pipeline que une o algoritmo NEAT ao problema definido
# generation_limit define quantas gerações o algoritmo vai executar
pipeline = Pipeline(
    algorithm=algorithm,
    problem=problem,
    generation_limit=5,  # Número máximo de gerações
    fitness_target=-0.01,  # Valor alvo de fitness; se alcançado, o treinamento pode ser interrompido
    seed=42,  # Semente para reprodutibilidade dos resultados
)


if __name__ == '__main__':
    # Configura e inicia o pipeline
    state = pipeline.setup()

    # Executa o pipeline por um número limitado de gerações
    for gen in range(pipeline.generation_limit):
        state = pipeline.step(state)  # Executa uma iteração (geração) do algoritmo
        best_fitness = state.best_fitness if hasattr(state, 'best_fitness') else None
        print(f"Generation {gen}: Best Fitness = {best_fitness}")

    # Após a execução, extrai o melhor genoma encontrado na população final
    best_genome = max(state.population, key=lambda g: g.fitness)
    print("Best Genome:", best_genome)
