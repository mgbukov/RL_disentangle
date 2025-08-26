import torch


if __name__ == "__main__":
    gpu_ok = False

    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True

    if not gpu_ok:
        print(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    else:
        print("GPU ok!")
