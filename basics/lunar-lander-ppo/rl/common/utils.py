import torch


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a tensor to zero mean and unit variance.

    Used to normalize returns or advantages before computing policy gradient.
    Reduces variance in the gradient estimates.

    *Why normalize returns?*

    Raw returns can vary wildly in scale. One episode might have returns in the range
    [50, 200], another [-100, 50]. THis makes the gradient magnitude inconsistent -
    big returns = big gradients = unstable learning.

    Normalize to mean=0, std=1 keeps the gradient scale consistent across batches. The relative
    ranking of actions is preserved (good actions still have higher-than-average returns), but
    the absolute scale is tamed.

    :param x - Input tensor
    :param eps - Small constant to avoid division by zero. If all returns happen to be identical,
        (std=0), we'd divide by zero. Adding a tiny epsilon prevents that edge case.
    """

    # Use unbiased=False for small batches (avoids division by n-1 which fails for n=1)
    return (x - x.mean()) / (x.std(unbiased=False) + eps)
