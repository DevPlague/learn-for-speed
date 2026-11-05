"""Turns a saved PPO model into a policy callable"""
from stable_baselines3 import PPO

def load_policy(path):
    """Observation in, deterministic action out. The reward is normalized during training."""
    model = PPO.load(path, device="cpu")

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy
