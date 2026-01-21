import torch
import torch.nn as nn  # Pytorch's neural network module
from torch.distributions import Categorical  # A probability distribution over discrete choices.

# This is what turns our logits into something we can sample from and compute log_probs with.


def mlp(sizes: list[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Takes a list like [8, 64, 64, 4] and builds:
    Linear(8->64) -> Tanh -> Linear(64->64) -> Tanh -> Linear(64->4)

    The final layers output is logits - raw scores that can be positive or negative.
    Softmax (applied later in Categorical) expects unbounded inputs. If we put Tanh there,
    we'd squash everything to [-1, 1] before softmax, limiting expressiveness.

    For policy gradients, Tanh is a common default over ReLU. It's bounded and smooth.

    :param sizes - List of layer sizes, e.g. [8, 64, 64, 4]
    :param activation - Activation function between layers (not after final layer)
    """

    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:  # No activation after final layer
            layers.append(activation())
    return nn.Sequential(*layers)


class CategoricalPolicy(nn.Module):
    """Policy network for discrete action spaces."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]) -> None:
        """Builds the network.

        If obs_dim=8, hidden_sizes=[64,64], act_dim=4:
            mlp([8, 64, 64, 4]) -> Linear(8->64) -> Tanh -> Linear(64->64) -> Tanh -> Linear(64->4)

        :param obs_dim - Dimensionality of the observation space
        :param act_dim - Dimensionality of the action space
        :param hidden_sizes - List of hidden layer sizes
        """
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes + [act_dim])

    def forward(self, obs: torch.Tensor) -> Categorical:
        """Return action distribution for given observations.


        Takes observations, returns a Categorical distribution. We don't sample here -
        jsut build the distrubtion. This separation is useful because sometimes you want
        the distribution itself (for computing entropy, evaluating actions, etc.).
        :param obs - Observation tensor
        """
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[int, float]:
        """Sample an action and return (action, log_prob).

        Convenience method for training loop.
        - Stochastic (deterministic=False): Sample from the distribution. Used during training
            for exploration.
        - Deterministic (deterministic=True): Take the highest probability action (argmax). Used
            during evaluation.



        :param obs - Observation tensor
        :param deterministic - Whether to sample deterministically
        :return (action, log_prob) - Python scalars. The training loop needs the action to send to the
            env, and we store log_prob in the buffer
        """
        dist = self.forward(obs)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        # Why .item()? Converts single-element tensors to Python numbers. The env expects a plain int,
        # not a tensor.
        return int(action.item()), float(log_prob.item())

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_probs and entropy for given obs-action pairs.

        Used during update to recompute log_probs with current policy.

        We need this because during the update, we have a batch of (obs, action)
        pairs from the buffer and we need to:
            - *Recompute logprobs* with the current policy weights. For REINFORCE
                this gives the same answer (policy hasn't changed). For PPO, the
                policy changes between collection and update, so we need both old (stored)
                and new (recomputed) log_probs.
            - *Compute entropy* for the entropy bonus. Entropy measures "how random is the
                policy?" High entropy = exploring, low_entropy = confident/ exploiting. We
                often add a smal entropy bonus to the loss to encourage exploration:
                    loss = policy_loss - entropy_coef * entropy
                    (minus because we want to maximize entropy, but we're minimizing loss)

        :param obs - Observation tensor
        :param actions - Action tensor
        :return (log_probs, entropy) - Tensors of shape (batch_size,). If you pass a batch of
            100 observations and 100 actions, you get back 100 log_probs and 100 entropy values.
        """

        dist = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class Critic(nn.Module):
    """Value network that estimates V(s) - expected return from a state."""

    def __init__(self, obs_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        # no act_dim param. The output is always 1 (a single value estimate).
        self.net = mlp([obs_dim] + hidden_sizes + [1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # The squeeze(-1) removes the last dimension. If we pass a batch of 32 observations:
        # - self.net(obs) returns shape (32, 1)
        # - .squeeze(-1) makes it (32,)
        return self.net(obs).squeeze(-1)
