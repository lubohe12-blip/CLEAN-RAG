import torch


def get_device(cfg):
    prefer = cfg["runtime"]["device"]

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")