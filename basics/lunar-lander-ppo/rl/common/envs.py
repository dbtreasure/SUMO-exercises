import gymnasium as gym


def make_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    """Create a Gymnasium environment.

    Seeding is handled separately via env.reset(seed=...) and seed_everything().
    """
    return gym.make(env_id, render_mode=render_mode)
