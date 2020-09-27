import numpy as np
from RlGlue import BaseEnvironment

# Constants
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]

class CliffWorld(BaseEnvironment):
    def __init__(self, shape=[8, 5]):
        self.shape = shape
        self.state = [0, 0]

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

        # check if fallen in cliff
        is_first_row = self.state[1] == 0
        is_last_col = self.state[0] == self.shape[0] - 1
        is_first_col = self.state[0] == 0
        is_not_outside_cols = not is_last_col and not is_first_col

        if is_first_row and is_not_outside_cols:
            reward = -100
            terminal = True
        elif is_first_row and is_last_col:
            reward = 1
            terminal = True
        else:
            reward = 0
            terminal = False

        state_idx = np.ravel_multi_index(self.state, self.shape)
        return (reward, state_idx, terminal)