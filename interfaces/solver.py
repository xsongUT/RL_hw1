import gymnasium as gym

from interfaces.policy import Policy

class Hyperparameters:
    def __init__(self, gamma: float):
        self.gamma = gamma
        """Controls the level of discounting, usual values are in [0.95, 1.0]"""

class Solver(Policy):
    def __init__(self, name: str, env: gym.Env, hyperparameters: Hyperparameters):
        self.name = name
        self.env = env
        self.hyperparameters = hyperparameters

        self.writer = None
    
    def train_episode(self):
        raise NotImplementedError
    
    def eval_episode(self) -> float:
        G = 0.0
        state, _ = self.env.reset()
        done = False

        while not done: 
            action = self.action(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            G += reward
            done = terminated or truncated

        return G
    
    def action(self, state):
        raise NotImplementedError
    
    def action_prob(self, state, action):
        raise NotImplementedError
    