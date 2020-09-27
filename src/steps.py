import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from multiprocessing.pool import Pool
from functools import partial
import matplotlib.pyplot as plt

from RlGlue import RlGlue

from agents.QLearning import QLearning
from agents.SoftmaxQ import SoftmaxQ
from environments.CliffWorld import CliffWorld
from environments.PotholeWorld import PotholeWorld

num_steps = 5000
runs = 1000
SHAPE = [10, 10]

def test_algorithm(agent, run):
    # set random seeds accordingly
    np.random.seed(run)

    env = PotholeWorld(SHAPE)

    glue = RlGlue(agent, env)
    agent.reset()

    # Run the experiment
    rewards = []
    glue.start()
    for _ in range(num_steps):
        r, o, a, t = glue.step()
        rewards.append(r)

        if t:
            glue.start()

    return rewards


if __name__ == "__main__":
    pool = Pool()

    q_agent = QLearning(states=np.prod(SHAPE), actions=4)
    q_rewards = pool.map(partial(test_algorithm, q_agent), range(runs))

    print('done q_agent')

    softmax_agent = SoftmaxQ(states=np.prod(SHAPE), actions=4)
    softmax_rewards = pool.map(partial(test_algorithm, softmax_agent), range(runs))

    print('done softmax_agent')

    plt.plot(np.mean(q_rewards, axis=0), label='Q-learning', color='blue')
    plt.plot(np.mean(softmax_rewards, axis=0), label='Softmax Q', color='red')
    plt.legend()
    plt.show()