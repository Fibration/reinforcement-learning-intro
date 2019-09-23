import random


# slot machine game for bandits
class Game:
    def __init__(self):
        self.arms = [random.uniform(-5, 5) for i in range(10)]

    def play(self, arm: int) -> float:
        if arm < 9:
            return random.normalvariate(self.arms[arm], 1)


