import tensorflow as tf
from tensorflow.python.client import device_lib

print("TensorFlow version:", tf.__version__)
print("CUDA support built:", tf.test.is_built_with_cuda())

print("Available devices:")
print(device_lib.list_local_devices())

print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

# Updated code to get cuDNN version
cuda_version = tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')
cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')
print("Updated CUDA version:", cuda_version)
print("Updated cuDNN version:", cudnn_version)