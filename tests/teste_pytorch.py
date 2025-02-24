import torch

def test_pytorch_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.rand(3, 3).to(device)
        y = torch.rand(3, 3).to(device)
        z = x + y
        print("PyTorch is using GPU acceleration.")
        print("Result of tensor addition on GPU:", z)
    else:
        print("GPU is not available. PyTorch is using CPU.")

if __name__ == "__main__":
    test_pytorch_gpu()