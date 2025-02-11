import torch
print("CUDA está disponível:", torch.cuda.is_available())
print("Número de GPUs:", torch.cuda.device_count())
print("Nome da GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma GPU detectada")
