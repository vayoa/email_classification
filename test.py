import torch, platform

print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
