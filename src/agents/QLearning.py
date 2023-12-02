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
        uniform = np.ones(self.actions) / self.actions
        greedy = np.zeros(self.actions)
        greedy[random.argmax(self.Q[s])] = 1

        return self.epsilon * uniform + (1 - self.epsilon) * greedy

    def _selectAction(self, s):
        return random.sample(self._policy(s))

    def start(self, s):
        self.s_t = s
        self.a_t = self._selectAction(s)
        return self.a_t

    def step(self, r, s):
        delta = (r + self.gamma * np.max(self.Q[s])) - self.Q[self.s_t, self.a_t]
        self.Q[self.s_t, self.a_t] += self.alpha * delta

        self.s_t = s
        self.a_t = self._selectAction(s)

        return self.a_t

    def end(self, r):
        delta = r - self.Q[self.s_t, self.a_t]
        self.Q[self.s_t, self.a_t] += self.alpha * delta

    def reset(self):
        self.Q[:] = 0
        self.s_t = None
        self.a_t = None
