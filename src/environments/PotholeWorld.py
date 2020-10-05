import numpy as np
from RlGlue import BaseEnvironment

# Constants
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]

class PotholeWorld(BaseEnvironment):
    def __init__(self, shape=[10, 10], pothole_perc=0.25):
        self.shape = shape
        self.state = [0, 0]
        self.holes = []

        for i in range(shape[0]):
            for j in range(shape[0]):
                eps = np.random.rand()
                if eps > pothole_perc or i == shape[0] - 1 and j == shape[1] - 1:
                    continue

                self.holes.append((i, j))

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
        if tuple(self.state) in self.holes:
            reward = -10

        elif self.state[0] == self.shape[0] - 1 and self.state[1] == self.shape[1] - 1:
            terminal = True
            reward = 100

        state_idx = np.ravel_multi_index(self.state, self.shape)
        return (reward, state_idx, terminal)
