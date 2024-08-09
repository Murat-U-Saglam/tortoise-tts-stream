import torch

if torch.cuda.is_available():
    print("CUDA is installed.")
else:
    print("CUDA is not installed.")