import numpy as np
from RlGlue import BaseEnvironment

# Constants
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]

class PotholeWorld(BaseEnvironment):
    def __init__(self, shape=[10, 10]):
        self.shape = shape
        self.state = [0, 0]
        self.hole = [ int(self.shape[0] // 2), int(self.shape[1] // 2) ]

    def start(self):
        self.state = [0, 0]
        state_idx = np.ravel_multi_index(self.state, self.shape)
        return state_idx

    def step(self, a):
        if a == RIGHT:
            self.state[0] = min(self.state[0] + 1, self.shape[0] - 1)
        elif a == LEFT:
            self.state[0] = max(0, self.state[0] - 1)
        elif a == UP:
            self.state[1] = min(self.state[1] + 1, self.shape[1] - 1)
        elif a == DOWN:
            self.state[1] = max(0, self.state[1] - 1)
        else:
            raise Exception("Unknown action given: " + str(a))


        terminal = False
        reward = -0.1
        if self.state[0] == self.hole[0] and self.state[1] == self.hole[1]:
            reward = -10

        elif self.state[0] == self.shape[0] - 1 and self.state[1] == self.shape[1] - 1:
            terminal = True
            reward = 1

        state_idx = np.ravel_multi_index(self.state, self.shape)
        return (reward, state_idx, terminal)