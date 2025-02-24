import tensorflow as tf
import time

# Criar um tensor grande
a = tf.random.normal([10000, 10000])
b = tf.random.normal([10000, 10000])

# Testar no CPU
with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    end = time.time()
    print(f"Tempo no CPU: {end - start:.4f} segundos")

# Testar na GPU
with tf.device('/GPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    end = time.time()
    print(f"Tempo na GPU: {end - start:.4f} segundos")
