import numpy as np
from typing import List, Tuple, Union
import jax
import jax.numpy as jnp

from tensorneat.problem.supervised import SupervisedFuncFit

# Definindo função auxiliar para extrair a saída final
def get_final_output(raw_output):
    """
    Extrai a saída final da rede neural.
    Se a saída for uma tupla, assume que o segundo elemento é a saída final.
    """
    if isinstance(raw_output, (tuple, list)):
        # Se for uma tupla com 4 ou mais elementos, assumimos que o segundo é a saída final
        if len(raw_output) >= 4:
            return raw_output[1]
        else:
            return raw_output[0]
    return raw_output

def softmax(x):
    """Função de ativação Softmax usando JAX."""
    exp_x = jnp.exp(x - jnp.max(x))  # Para evitar overflow numérico
    return exp_x / jnp.sum(exp_x)

# Definição da classe do problema supervisionado:

class CustomSupervisedFuncFit(SupervisedFuncFit):
    def __init__(
            self, 
            X: Union[List, Tuple, np.ndarray], 
            y: Union[List, Tuple, np.ndarray], 
            batch_size: int = 256,
            *args,
            **kwargs):  # Processamento em lotes para reduzir consumo de memória)
        """
        Problema de aprendizado supervisionado para TensorNEAT.

        X: Features (inputs)
        y: Labels (outputs, one-hot encoded)
        batch_size: Tamanho do batch para avaliação (evita estouro de memória)
        """

        self.data_inputs = jnp.array(X, dtype=jnp.float32)
        self.data_outputs = jnp.array(y, dtype=jnp.float32)
        self.batch_size = batch_size

        super().__init__(X, y, *args, **kwargs)

    # @property é um decorador que permite chamar um método como um atributo por exemplo ao invés de chamar supervised_problem.inputs() você pode chamar supervised_problem.inputs.
    @property
    def inputs(self):
        """
        Retorna os inputs (X_train).
        """
        return self.data_inputs  # Retorna os inputs (X_train)

    @property
    def targets(self):
        """
        Retorna os targets (y_train).
        """
        return self.data_outputs  # Retorna os labels (y_train)

    @property
    def input_shape(self):
        """
        Retorna a forma esperada da entrada.
        O TensorNEAT espera um formato (features,).
        """
        return (self.data_inputs.shape[1],)  # Retorna o número de features

    @property
    def output_shape(self):
        """
        Retorna a forma esperada da saída.
        O TensorNEAT espera um formato (classes,).
        """
        return self.data_outputs.shape  # Retorna o shape dos labels (one-hot encoding)

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
	
        num_samples = self.data_inputs.shape[0]
        num_batches = max(1, num_samples // self.batch_size)  # Evita divisão por zero
        total_loss = 0.0

        # Processamos os dados em batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, num_samples)

            batch_X = self.data_inputs[batch_start:batch_end]
            batch_y = self.data_outputs[batch_start:batch_end]
            
            subkey, _ = jax.random.split(randkey)
            perturbed_X = batch_X + 0.01 * jax.random.normal(subkey, batch_X.shape)  # Pequena perturbação (opcional)

            # Aplicar act_func para cada entrada e empilhar as saídas finais
            predictions_list = []
            for i, x in enumerate(perturbed_X):
                print(x)
                raw_output = act_func(state, x, self, params)
                out = get_final_output(raw_output)
                predictions_list.append(out)

            predictions = jnp.stack(predictions_list)
            batch_loss = -jnp.mean(jnp.sum(batch_y * jnp.log(predictions + 1e-9), axis=1))
            total_loss += batch_loss
        
        return -total_loss / num_batches