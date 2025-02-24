import jax
import jax.numpy as jnp
from tensorneat.common import ACT, AGG

def feed_forward_fn(genome, inputs):
    # Decodifica o genome para uma rede
    net = genome.decode(
        activation=ACT.RELU,   # ou outra se preferir
        aggregator=AGG.sum,    # ou outro tipo de agregação
        use_bias=True
    )
    # Roda a rede nas entradas
    outputs = net.run(inputs)  # shape [batch_size, num_outputs]
    return outputs

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

def evaluate(genome, inputs, batch_size, randkey):
    """
        Calcula a função de fitness usando Cross-Entropy Loss em batches.
        
        Retorna:
            - Fitness (negativo da perda média)
    """

    num_samples = inputs[0].shape[0]
    num_batches = max(1, num_samples // batch_size)  # Evita divisão por zero
    total_loss = 0.0

    # Processamos os dados em batches
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)

        batch_X = inputs[0][batch_start:batch_end]
        batch_y = inputs[0][batch_start:batch_end]
        
        subkey, _ = jax.random.split(randkey)
        perturbed_X = batch_X + 0.01 * jax.random.normal(subkey, batch_X.shape)  # Pequena perturbação (opcional)

        # Aplicar act_func para cada entrada e empilhar as saídas finais
        predictions_list = []
        for x in perturbed_X:
            raw_output = feed_forward_fn(genome, x)
            predictions_list.append(raw_output)

        predictions = jnp.stack(predictions_list)
        batch_loss = -jnp.mean(jnp.sum(batch_y * jnp.log(predictions + 1e-9), axis=1))
        total_loss += batch_loss
    
    return -total_loss / num_batches