import numpy as np
from scipy.optimize import brentq

from src.agents.QLearning import QLearning
from src.utils import random

def generateF(vals, mm):
    def f(x):
        exps = np.exp(x * (vals - mm))
        return np.sum(exps * (vals - mm))

    return f

class MellowMaxQ(QLearning):
    def __init__(self, states, actions):
        super().__init__(states, actions)
        self.tau = 3.0

    def _policy(self, s):
        actions = self.Q[s]
        n = len(actions)

        m = np.max(actions)
        exps = np.exp(self.tau * (actions - m))
        mm = m + np.log(np.sum(exps) / n) / self.tau

        f = generateF(actions, mm)

        # in a few instances, numerical stability can cause this to fail
        # when that happens, just revert back to a softmax policy
        try:
            beta = brentq(f, 0, 1000)
        except:
            beta = 1

        exps = np.exp(beta * (actions - m))
        probs = exps / np.sum(exps)

        return probs
