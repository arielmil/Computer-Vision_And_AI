
import os
import sys

# Seta a variavel de ambiente para o tensorflow usar a alocação de memória assíncrona
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Pega o path absoluto do diretorio do arquivo
curPath = os.path.dirname(os.path.abspath(__file__))

# Vai para o diretorio de cima (Computer_vision)
Computer_vision_path = os.path.abspath(os.path.join(curPath, os.pardir))

# Adiciona o path do diretorio do arquivo ao sys.path
sys.path.append(Computer_vision_path + "/NEAT-Algorithms")

from dataset_manager import load_images_from_folder
from dataset_transformer import images_dir
import autokeras as ak
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow.keras.optimizers as optimizers # type: ignore
import tensorflow as tf

tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.chdir(curPath)

X_data, y_data = [], []
for X_part, y_part in load_images_from_folder(images_dir, everything_at_once=False, flatten = False):
    X_data.append(X_part)
    y_data.append(y_part)

# Concat all partial data
X_data = np.concatenate(X_data, axis=0)
y_data = np.concatenate(y_data, axis=0)

X_data = X_data.reshape(-1, 20, 20, 1)  # Adiciona canal extra para grayscale
if len(y_data.shape) > 1:  # Se for one-hot, converte para rótulos inteiros
    y_data = np.argmax(y_data, axis=1)

# If your dataset_manager gives integer labels 0..9, 
# but your code snippet shows:
#       y_data = np.eye(10)[y_data]
# That’s one-hot. Typically for cross-entropy, we don’t want one-hot in training. 
# However, if you *did* create one-hot above, let's keep it for now 
# (and in eval_genomes, we call .argmax). 
#
# If you never turned them into one-hot, no problem, just keep them as integer class labels.

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, 
    y_data, 
    test_size=0.3, 
    random_state=42
)

batch_size = 8

print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,
    tuner="greedy"
)

optimizer = optimizers.Adam(learning_rate=1e-3)

clf.fit(X_train, y_train, epochs = 30)

print("Avaliando modelo...")
print("Label da primeira entrada:")
print(y_test[0])
print("Label da segunda entrada:")
print(y_test[1])
print("Label da terceira entrada:")
print(y_test[2])

print("Label da terceira pra ultima entrada:")
print(y_test[-3])
print("Label da penultima entrada:")
print(y_test[-2])
print("Label da ultima entrada:")
print(y_test[-1])

results = clf.predict(X_test)

# Salva o melhor modelo encontrado
best_model = clf.export_model()

# Pega o diretorio pai do diretorio atual
model_dir = os.path.abspath(os.path.join(Computer_vision_path, "neural-networks", "models"))

model_path = os.path.join(model_dir, "melhor_modelo_autokeras.keras")

best_model.save(model_path)  # Salva no formato .keras

print(results)