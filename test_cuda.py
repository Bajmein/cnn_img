import torch

print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA device count:", torch.cuda.device_count())
print("Torch CUDA current device:", torch.cuda.current_device())
print("Torch CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")