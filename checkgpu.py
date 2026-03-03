import torch

if torch.cuda.is_available():
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    print(f"{torch.cuda.get_device_name()}")
else:
    print("CUDA is not available. PyTorch is using the CPU.")
    