import os
import kagglehub
import numpy as np
import pickle
from PIL import Image
from typing import Union, List, Tuple

from sklearn.model_selection import train_test_split

from tensorneat.problem.supervised import SupervisedFuncFit
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome, DefaultMutation, DefaultCrossover, DefaultDistance, DefaultConn
from tensorneat.common import ACT, AGG
from tensorneat.genome.gene.node.bias import BiasNode
from tensorneat.problem.func_fit import FuncFit

import jax
from jax import numpy as jnp, vmap, random
import numpy as np

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
        
        Parâmetros:
            - state: Estado atual do algoritmo (necessário para o TensorNEAT)
            - randkey: Chave aleatória do JAX (necessário para inicializações ou mutações)
            - act_func: Função de ativação da rede neural
            - params: Parâmetros da rede (pesos e conexões)
        
        Retorna:
            - Fitness (negativo da perda média)
        """

        # Se state for None, definir para array vazio para evitar erros
        if state is None:
            state = jnp.array([])

        # Definindo função auxiliar para extrair a saída final
        def get_final_output(raw_output):
            if isinstance(raw_output, (tuple, list)):
                # Se for uma tupla com 4 ou mais elementos, assumimos que o segundo é a saída final
                if len(raw_output) >= 4:
                    return raw_output[1]
                else:
                    return raw_output[0]
            return raw_output

        num_samples = self.data_inputs.shape[0]
        num_batches = max(1, num_samples // self.batch_size)  # Evita divisão por zero
        total_loss = 0.0

        # Processamos os dados em batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, num_samples)

            batch_X = self.data_inputs[batch_start:batch_end]
            batch_y = self.data_outputs[batch_start:batch_end]

            # Gerando variação randômica baseada no randkey (pode ser usada para dropout ou noise nos dados)
            subkey, _ = jax.random.split(randkey)
            perturbed_X = batch_X + 0.01 * jax.random.normal(subkey, batch_X.shape)  # Pequena perturbação (opcional)
            
            print(f"Printando antes do vmap e antes de perturbed_x passar por jnp.reshape:")
            print(f"🚨 Dentro de evaluate - batch_X.shape: {batch_X.shape}")
            print(f"🚨 Dentro de evaluate - esperado: ({self.batch_size}, 12600)")
            print(f"🚨 Dentro de evaluate - act_func antes: {act_func}")
            print(f"🚨 Dentro de evaluate - state.shape: {state.shape if isinstance(state, jnp.ndarray) else 'None'}")

            print(f"📌 Shape antes do vmap: {perturbed_X.shape}")
            print(f"Printando antes do vmap e depois de perturbed_x passar por jnp.reshape:")
            print(f"🚨 Dentro de evaluate - batch_X.shape: {batch_X.shape}")
            print(f"🚨 Dentro de evaluate - esperado: ({self.batch_size}, 12600)")
            print(f"🚨 Dentro de evaluate - act_func antes: {act_func}")
            print(f"🚨 Dentro de evaluate - state.shape: {state.shape if isinstance(state, jnp.ndarray) else 'None'}")
            
            print(f"📌 Antes de reshape - perturbed_X.shape: {perturbed_X.shape}")
            if len(perturbed_X.shape) > 2:
                perturbed_X = perturbed_X.reshape(perturbed_X.shape[0], -1)
            else:
                perturbed_X = perturbed_X.reshape(self.batch_size, -1)
            print(f"📌 Depois de reshape - perturbed_X.shape: {perturbed_X.shape}")

            assert perturbed_X.shape[1] == 12600, f"Formato incorreto: {perturbed_X.shape}"
            assert batch_X.shape[1] == 12600, f"Formato incorreto: {batch_X.shape}"
            
            print(f"📌 Shape antes do vmap: {perturbed_X.shape}")
            for i, x in enumerate(perturbed_X):
                print(f"🚨 Teste direto - Entrada {i} (shape): {x.shape}")
                if x.shape != (12600,):
                    print(f"❌❌❌ ERRO: Entrada {i} tem shape inválido: {x.shape} (esperado: (12600,)) ❌❌❌")
                    exit(1)
            
            print(f"Perturbed_X.shape: {perturbed_X.shape}")
            print(f"Perturbed_X[0].shape: {perturbed_X[0].shape}")
            print(f"Params: {params}")

            # Aplicar act_func para cada entrada e empilhar as saídas finais
            predictions_list = []
            for i, x in enumerate(perturbed_X):
                raw_output = act_func(state, x[None, :], params)
                print(f"🚨 Debug: Raw output for entrada {i}: {raw_output} (type: {type(raw_output)})")
                if isinstance(raw_output, (tuple, list)):
                    print(f"🚨 Debug: Length of raw output for entrada {i}: {len(raw_output)}")
                out = get_final_output(raw_output)
                predictions_list.append(out)
            
            predictions = jnp.stack(predictions_list)
            
            print(f"📌 Shape após vmap: {predictions.shape}")
            
            batch_loss = -jnp.mean(jnp.sum(batch_y * jnp.log(predictions + 1e-9), axis=1))
            total_loss += batch_loss
        
        return -total_loss / num_batches

    @property
    def inputs(self):
        return self.data_inputs  # 🔹 Retorna os inputs (X_train)

    @property
    def targets(self):
        return self.data_outputs  # 🔹 Retorna os labels (y_train)

    @property
    def input_shape(self):
        """
        Retorna a forma esperada da entrada.
        O TensorNEAT espera um formato (features,).
        """
        return (self.data_inputs.shape[1],)  # Retorna o número de features

    @property
    def output_shape(self):
        return self.data_outputs.shape  # 🔹 Retorna o shape dos labels (one-hot encoding)

def load_images_from_folder(folder_path, everything_at_once=False):
    """
    Carrega imagens do dataset em batches para evitar consumo excessivo de memória.
    """
    data, labels = [], []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            num_images = int(len(images))
            for image_file in images[:num_images]:
                img_path = os.path.join(label_path, image_file)
                img = Image.open(img_path).convert('L')  # Escala de cinza
                img = img.resize((90, 140))  # Redimensiona
                img = (np.array(img) / 255.0).flatten()  # Normaliza
                
                data.append(img)
                labels.append(int(label))

                if not everything_at_once:
                    # 🔹 Se a memória estiver alta, libere os dados periodicamente
                    if len(data) % 5000 == 0:
                        print(f"🔹 Carregadas {len(data)} imagens, liberando memória...")
                        yield np.array(data), np.array(labels)
                        data, labels = [], []

    if everything_at_once:
        # 🔹 Retorna todos os dados de uma só vez
        yield np.array(data), np.array(labels)
    else:
        # 🔹 Retorna o restante dos dados
        yield np.array(data), np.array(labels)

def softmax(x):
    """Função de ativação Softmax usando JAX."""
    exp_x = jnp.exp(x - jnp.max(x))  # Para evitar overflow numérico
    return exp_x / jnp.sum(exp_x)

# Caminho para o dataset
path = r'/home/mileguir/.cache/kagglehub/datasets/olafkrastovski/handwritten-digits-0-9/versions/2'
if not os.path.exists(path):
    print(f"Dataset não encontrado em {path}. Baixando...")
    path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
else:
    print(f"✅ Dataset encontrado em {path}.")

# Carregar o dataset
X_data, y_data = [], []
for X_part, y_part in load_images_from_folder(path, everything_at_once=False):
    X_data.append(X_part)
    y_data.append(y_part)

# 🔹 Concatena os lotes de forma eficiente
X_data = np.concatenate(X_data, axis=0)
y_data = np.concatenate(y_data, axis=0)

# Converter labels para one-hot encoding
y_data = np.eye(10)[y_data]

# Dividir dataset em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
print(f"🔹 Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# Criar problema de aprendizado supervisionado
supervised_problem = SupervisedFuncFit(X_train, y_train, batch_size=32)

# Configura a arquitetura da rede neural
genome = DefaultGenome(
    num_inputs=12600,  # Número de pixels da imagem (entrada)
    num_outputs=10,  # 10 classes (0-9)
    max_nodes=13000, # Número máximo de neurônios
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
