def load_config(config_path):
    import yaml

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_device():
    import torch

    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
