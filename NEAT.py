import os
import kagglehub
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorneat.problem.supervised import SupervisedFuncFit
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT, AGG
from tensorneat.genome.gene.node.bias import BiasNode
from typing import Union, List, Tuple
from jax import numpy as jnp, vmap
import jax.numpy as jnp
import numpy as np
from tensorneat.problem.func_fit import FuncFit

class SupervisedFuncFit(FuncFit):
    def __init__(
        self,
        X: Union[List, Tuple, np.ndarray],
        y: Union[List, Tuple, np.ndarray],
        batch_size: int = 256,  # 🔹 Processamento em lotes para reduzir consumo de memória
        *args,
        **kwargs,
    ):
        """
        Problema de aprendizado supervisionado para TensorNEAT.

        X: Features (inputs)
        y: Labels (outputs, one-hot encoded)
        batch_size: Tamanho do batch para avaliação (evita estouro de memória)
        """
        self.data_inputs = jnp.array(X, dtype=jnp.float32)
        self.data_outputs = jnp.array(y, dtype=jnp.float32)
        self.batch_size = batch_size

        super().__init__(*args, **kwargs)

    def evaluate(self, state, randkey, act_func, params):
        """
        Calcula a função de fitness usando Cross-Entropy Loss em batches.
        """
        num_samples = self.data_inputs.shape[0]
        total_loss = 0.0

        for i in range(0, num_samples, self.batch_size):
            batch_X = self.data_inputs[i : i + self.batch_size]
            batch_y = self.data_outputs[i : i + self.batch_size]
            
            print(f"🔍 batch_X.shape = {batch_X.shape}, esperado = ({self.batch_size}, {self.input_shape[1]})") # Deve ser (128, 12600)
            print(f"📌 Shape de um único x dentro de batch_X: {batch_X[0].shape}")  # Deve ser (12600,)

            for x in batch_X:
                print(f"📌 Shape real de x enviado para act_func: {x.shape}")
                print(f"📌 Shape após reshape: {x.reshape(1, -1).shape}")
                resultado = act_func(state, x.reshape(1, -1), params)  # Aqui testamos antes de rodar o vmap
                print(f"📌 Saída esperada de act_func: {resultado.shape}")

            # 🔹 Usa `act_func` corretamente dentro do NEAT para prever as saídas
            predictions = vmap(lambda x: act_func(state, x.reshape(1, -1), params))(batch_X)

            # 🔹 Calcula Cross-Entropy Loss no batch
            batch_loss = -jnp.mean(jnp.sum(batch_y * jnp.log(predictions + 1e-9), axis=1))
            total_loss += batch_loss

        return -total_loss / (num_samples / self.batch_size)  # 🔹 Fitness = -Loss

    @property
    def inputs(self):
        return self.data_inputs  # 🔹 Retorna os inputs (X_train)

    @property
    def targets(self):
        return self.data_outputs  # 🔹 Retorna os labels (y_train)

    @property
    def input_shape(self):
        return self.data_inputs.shape  # 🔹 Retorna o shape (n_amostras, n_features)

    @property
    def output_shape(self):
        return self.data_outputs.shape  # 🔹 Retorna o shape dos labels (one-hot encoding)

def load_images_from_folder(folder_path, use_percentage=1.0):
    """
    Carrega imagens do dataset, normaliza e converte para um formato adequado para TensorNEAT.
    use_percentage: Determina a porcentagem das imagens a serem usadas.
    """
    data, labels = [], []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            num_images = int(len(images) * use_percentage)
            for image_file in images[:num_images]:
                img_path = os.path.join(label_path, image_file)
                img = Image.open(img_path).convert('L')  # Converter para escala de cinza
                img = img.resize((90, 140))  # Ajustar para o tamanho correto (90x140)
                img = (np.array(img) / 255.0).flatten()  # Normalizar e achatar
                data.append(img)
                labels.append(int(label))
    
    return np.array(data), np.array(labels)


def softmax(x):
    """Função de ativação Softmax usando JAX."""
    exp_x = jnp.exp(x - jnp.max(x))  # Para evitar overflow numérico
    return exp_x / jnp.sum(exp_x)

# Caminho para o dataset
path = r'C:/Users/mileguir/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2'
if not os.path.exists(path):
    print(f"Dataset não encontrado em {path}. Baixando...")
    path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
else:
    print(f"✅ Dataset encontrado em {path}.")

# Carregar o dataset
X_data, y_data = load_images_from_folder(path)
print(f"✅ Dataset carregado com {len(X_data)} amostras!")

# Converter labels para one-hot encoding
y_data = np.eye(10)[y_data]

# Dividir dataset em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
print(f"🔹 Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# Criar problema de aprendizado supervisionado
supervised_problem = SupervisedFuncFit(X_train, y_train, batch_size=128)

# Configurar a arquitetura da rede neural
genome = DefaultGenome(
    num_inputs=12600,  # Número de pixels da imagem (entrada)
    num_outputs=10,  # 10 classes (0-9)
    max_nodes=13000, # Número máximo de neurônios
    max_conns = 250000, # Número máximo de conexões
    init_hidden_layers=(),  # Deixa o NEAT evoluir a estrutura oculta
    node_gene=BiasNode(
        activation_options=[ACT.sigmoid],  # Ativação Sigmoid nos neurônios ocultos
        aggregation_options=[AGG.sum, AGG.product],  # Opções de agregação
    ),
    output_transform=softmax,  # Softmax na saída para classificação multiclasse
)

# Configurar o algoritmo NEAT
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
