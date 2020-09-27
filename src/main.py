import numpy as np
import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from RlGlue import RlGlue

from agents.QLearning import QLearning
from agents.SoftmaxQ import SoftmaxQ
from environments.CliffWorld import CliffWorld

num_episodes = 50
runs = 1000

def test_algorithm(agent):
    all_rewards = []
    for run in range(runs):
        print(run)
        # set random seeds accordingly
        np.random.seed(run)

        env = CliffWorld()

        glue = RlGlue(agent, env)
        agent.reset()

        # Run the experiment
        rewards = []
        for episode in range(num_episodes):
            glue.runEpisode()
            r = glue.total_reward
            rewards.append(r)
            
            glue.total_reward = 0

        all_rewards.append(rewards)

    return all_rewards


q_agent = QLearning(states=40, actions=4)
q_rewards = test_algorithm(q_agent)

softmax_agent = SoftmaxQ(states=40, actions=4)
softmax_rewards = test_algorithm(softmax_agent)

plt.plot(np.mean(q_rewards, axis=0), label='Q-learning', color='blue')
plt.plot(np.mean(softmax_rewards, axis=0), label='Softmax Q', color='red')
plt.legend()
plt.show()