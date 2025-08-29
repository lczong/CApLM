import torch

def log_gpu_count():
    print("="*60)
    print("GPU Information")
    print("="*60)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using {gpu_count} GPU(s)")
        print(f"Current GPU: {current_device} - {device_name}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs available - using CPU")