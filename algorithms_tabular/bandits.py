import random
from typing import List
from matplotlib import pyplot as plt
import math


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
def greedy_bandit(steps: int, epsilon: float = 0, init_Q: List[float] = [0] * 100) -> List[float]:
    game = Game(100)
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


def ucb_bandit(steps: int, exploration: float) -> List[float]:
    game = Game(100)
    Q = [0] * 100
    # each action has been taken 0 times but 0.1 to avoid division by zero
    N = [0.1] * 100
    rewards = []

    if exploration <= 0:
        raise ValueError('Not a valid UCB exploration parameter value.')

    for step in range(1, steps + 1):
        ucb = [Q[a] + exploration * math.sqrt(math.log(step) / N[a])
               for a in range(len(Q))]
        best = ucb.index(max(ucb))
        N[best] += 1  # update the count of the action taken
        reward = game.play(best)
        Q[best] = (reward + (step - 1) * Q[best]) / step
        if step > 1:
            rewards.append(rewards[-1] * (step - 1) / step + reward / step)
        else:
            rewards.append(reward)

    return rewards


def gradient_bandit(steps: int, learning_rate: float) -> List[float]:
    arms = 100
    game = Game(arms)
    H = [0] * arms
    pi = [1] * arms
    rewards = [0]

    for step in range(1, steps + 1):
        best = random.choices(range(arms), weights=pi)[0]
        reward = game.play(best)
        H = [H[a] + learning_rate * (reward - rewards[-1]) * (1 - pi[a])
             if a == best
             else H[a] - learning_rate * (reward - rewards[-1]) * pi[a]
             for a in range(arms)
             ]
        expH = [math.exp(Ha) for Ha in H]
        partition_function = sum(expH)
        pi = [expHa / partition_function for expHa in expH]

        # update average reward
        if step > 1:
            rewards.append(rewards[-1] * (step - 1) / step + reward / step)
        else:
            rewards = [reward]

    return rewards


def main():
    rewards = []
    steps = 2000
    rewards.append(greedy_bandit(steps))
    rewards.append(greedy_bandit(steps, 0.01))
    rewards.append(greedy_bandit(steps, 0.005))
    rewards.append(greedy_bandit(steps, init_Q=[5] * 50))
    rewards.append(greedy_bandit(steps, init_Q=[4] * 50))
    rewards.append(ucb_bandit(steps, 0.1))
    rewards.append(ucb_bandit(steps, 1))
    rewards.append(gradient_bandit(steps, 0.1))
    rewards.append(gradient_bandit(steps, 0.3))

    labels = [
        'Greedy',
        '0.01-Greedy',
        '0.005-Greedy',
        'Optimistic expectations 5',
        'Optimistic expectations 4',
        'UCB exploration 0.1',
        'UCB exploration 1',
        'Gradient bandit learning rate 0.1',
        'Gradient bandit learning rate 0.3'
    ]
    plot_performance(rewards, labels)


if __name__ == '__main__':
    main()
