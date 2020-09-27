import numpy as np
from RlGlue import BaseAgent

from src.utils import random

class QLearning(BaseAgent):
    def __init__(self, states, actions):
        self.epsilon = 0.2
        self.gamma = 0.9
        self.alpha = 0.5

        self.states = states
        self.actions = actions
        self.Q = np.zeros((states, actions))
        
        self.s_t = None
        self.a_t = None

    def _policy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.actions)

        return random.argmax(self.Q[s])

    def start(self, s):
        self.s_t = s
        self.a_t = self._policy(s)
        return self.a_t

    def step(self, r, s):
        delta = (r + self.gamma * np.max(self.Q[s])) - self.Q[self.s_t, self.a_t]
        self.Q[self.s_t, self.a_t] += self.alpha * delta

        self.s_t = s
        self.a_t = self._policy(s)

        return self.a_t

    def end(self, r):
        delta = r - self.Q[self.s_t, self.a_t]
        self.Q[self.s_t, self.a_t] += self.alpha * delta

    def reset(self):
        self.Q[:] = 0
        self.s_t = None
        self.a_t = None
