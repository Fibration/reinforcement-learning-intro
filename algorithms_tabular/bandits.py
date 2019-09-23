import random
from typing import List
from matplotlib import pyplot as plt


# slot machine game for bandits
# means of normal distributed slots are randomly chosen
# from a uniform distribution
class Game:
    def __init__(self):
        self.arms = [random.uniform(-5, 5) for i in range(10)]

    def play(self, arm: int) -> float:
        if arm < 9:
            return random.normalvariate(self.arms[arm], 1)
        else:
            return random.normalvariate(self.arms[9], 1)
        # force action to always be valid


# plot performance
def plot_performance(rewards: List[float], title='Bandit Performance'):
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.title(title)
    plt.show()


# Greedy bandit
def greedy(steps: int):
    game = Game()
    Q = [0] * 10  # initialise value function
    rewards = []  # keep track of performance

    for step in range(1, steps + 1):
        best = Q.index(max(Q))
        reward = game.play(best)
        Q[best] = (reward + (step - 1) * Q[best]) / step
        rewards.push(rewards[-1] * (step - 1) / step + reward / step)

    # plot what we get
    plot_performance(rewards)
