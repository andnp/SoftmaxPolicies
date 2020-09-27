import numpy as np

from src.agents.QLearning import QLearning
from src.utils import random

class SoftmaxQ(QLearning):
    def __init__(self, states, actions):
        super().__init__(states, actions)
        self.tau = 1.0

    def _policy(self, s):
        actions = self.Q[s]
        exps = np.exp(actions) / self.tau
        probs = exps / np.sum(exps)

        return random.sample(probs)