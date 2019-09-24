import random
from typing import List
from matplotlib import pyplot as plt


# slot machine game for bandits
# means of normal distributed slots are randomly chosen
# from a uniform distribution
class Game:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.arms = [random.uniform(-5, 5) for i in range(num_arms)]

    def play(self, arm: int) -> float:
        if arm < self.num_arms - 1:
            return random.normalvariate(self.arms[arm], 1)
        else:
            return random.normalvariate(self.arms[self.num_arms - 1], 1)
        # force action to always be valid


# plot performance
def plot_performance(rewards: List[List[float]],
                     labels: List[str] = [],
                     title: str = 'Bandit Performance'):
    for index, reward in enumerate(rewards):
        if len(labels) == 0:
            plt.plot(reward, label=index)
        else:
            plt.plot(reward, label=labels[index])
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.title(title)
    plt.legend()
    plt.savefig('./bandit_performance/' + title + '.png')


# Greedy bandit
def greedy(steps: int, epsilon: float = 0, init_Q: List[float] = [0] * 50) -> List[float]:
    game = Game(50)
    Q = init_Q  # initialise value function
    rewards = []  # keep track of performance

    for step in range(1, steps + 1):
        best = 0
        if random.random() < epsilon:
            best = random.randrange(0, len(Q))
        else:
            best = Q.index(max(Q))
        reward = game.play(best)
        Q[best] = (reward + (step - 1) * Q[best]) / step
        if step > 1:
            rewards.append(rewards[-1] * (step - 1) / step + reward / step)
        else:
            rewards.append(reward)

    return rewards


def main():
    rewards = []
    steps = 1000
    rewards.append(greedy(steps))
    rewards.append(greedy(steps, 0.01))
    rewards.append(greedy(steps, 0.005))
    rewards.append(greedy(steps, init_Q=[5] * 50))

    labels = [
        'Greedy',
        '0.01-Greedy',
        '0.005-Greedy',
        'Optimistic expectations 5'
    ]
    plot_performance(rewards, labels)


if __name__ == '__main__':
    main()
