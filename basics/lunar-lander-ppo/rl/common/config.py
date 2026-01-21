from pathlib import Path

import yaml
from pydantic import BaseModel


class EnvConfig(BaseModel):
    id: str = "LunarLander-v2"


class TrainConfig(BaseModel):
    total_steps: int = 500_000
    eval_every_steps: int = 10_000
    eval_episodes: int = 10
    log_every_steps: int = 1_000


class ModelConfig(BaseModel):
    hidden_sizes: list[int] = [64, 64]
    activation: str = "tanh"


class AlgoConfig(BaseModel):
    name: str = "reinforce"
    gamma: float = 0.99
    lr_actor: float = 0.001
    lr_critic: float = 0.001
    normalize_returns: bool = True
    entropy_coef: float = 0.0
    max_grad_norm: float | None = None


class Config(BaseModel):
    seed: int = 42
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    algo: AlgoConfig = AlgoConfig()


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
