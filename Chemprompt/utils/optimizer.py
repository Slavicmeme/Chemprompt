import torch

DEFAULT_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adamw": torch.optim.AdamW
}

def get_optimizer(optimizer_name="adam", parameters=None, learning_rate=0.001, **kwargs):
    """
    Returns an optimizer based on the given name.

    Args:
        optimizer_name (str): Name of the optimizer to use.
        parameters (iterable): Model parameters to optimize.
        learning_rate (float): Learning rate for the optimizer.
        kwargs: Additional arguments for specific optimizers.

    Returns:
        torch.optim.Optimizer: An instance of the optimizer.
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name in DEFAULT_OPTIMIZERS:
        return DEFAULT_OPTIMIZERS[optimizer_name](parameters, lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Available optimizers: {list(DEFAULT_OPTIMIZERS.keys())}")
