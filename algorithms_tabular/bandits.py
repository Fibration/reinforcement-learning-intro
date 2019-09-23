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


# plot performance
def plot_performance(rewards: List[float], title='Bandit Performance'):
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.title(title)
    plt.show()
