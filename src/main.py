import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from functools import partial
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt

from RlGlue import RlGlue

from agents.QLearning import QLearning
from agents.SoftmaxQ import SoftmaxQ
from agents.MellowMax import MellowMaxQ
from environments.CliffWorld import CliffWorld
from environments.PotholeWorld import PotholeWorld

num_episodes = 50
runs = 100

def test(agent, run):
    # set random seeds accordingly
    np.random.seed(run)

    env = PotholeWorld()

    glue = RlGlue(agent, env)
    agent.reset()

    # Run the experiment
    rewards = []
    for episode in range(num_episodes):
        glue.runEpisode()
        r = glue.total_reward
        rewards.append(r)

        glue.total_reward = 0

    return rewards

def test_algorithm(agent):
    print(agent.__class__.__name__)
    pool = Pool()
    return pool.map(partial(test, agent), range(runs))

if __name__ == "__main__":
    mellowmax_agent = MellowMaxQ(states=100, actions=4)
    mellowmax_rewards = test_algorithm(mellowmax_agent)

    q_agent = QLearning(states=100, actions=4)
    q_rewards = test_algorithm(q_agent)

    softmax_agent = SoftmaxQ(states=100, actions=4)
    softmax_rewards = test_algorithm(softmax_agent)

    plt.plot(np.mean(q_rewards, axis=0), label='Q-learning', color='blue')
    plt.plot(np.mean(softmax_rewards, axis=0), label='Softmax Q', color='red')
    plt.plot(np.mean(mellowmax_rewards, axis=0), label='Mellowmax Q', color='purple')
    plt.legend()
    plt.show()
