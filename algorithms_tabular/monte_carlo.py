from typing import List, Dict, Tuple, Callable
import sys
sys.path.append('../')
from games.blackjack import Blackjack

# implement Monte Carlo algorithms for discrete action,
# discrete state, episodic games
# state s is a List[int]  (one hot encode if needed)
# action a is an int >= 0 corresponding to a fixed list of actions
# Q and pi are Dict[List[int], float]
# Game is a class with method play which accepts an action
# and returns a reward and a new state


class Game:
    def __init__(self, init: List[int] = [0, 0, 0, 0]):
        self.state = init
        self.reward = 0
        self.terminal = False

    def play(self, action: int) -> (float, List[int]):
        if self.terminal:
            raise ValueError('Cannot play past terminal state!')
        self.state[action] += 1
        reward = self.state[action] / sum(self.state)
        self.reward += reward
        self.terminal = self.check_terminal()
        return reward, self.state

    def get_state(self) -> List[int]:
        return self.state

    def check_terminal(self) -> bool:
        return sum(self.state) >= 18 or max(self.state) >= 9


def mc_prediction(game,
                  num_episodes: int,
                  pi: Callable[[Tuple[int]], int],
                  gamma: float) -> Dict[List[int], float]:
    # initialise Q and rewards
    returns = dict()
    Q = dict()

    for episode in range(num_episodes):
        curr_game = game()
        states = []
        rewards = []
        while not curr_game.terminal:
            state = curr_game.get_state()
            action = pi(state)
            states.append(tuple(state))
            reward, _ = curr_game.play(action)
            rewards.append(reward)

        # estimate Q from episode
        G = 0
        while len(states) != 0:
            G = gamma * G + rewards.pop()
            state = states.pop()
            if state not in set(states):
                if state in returns:
                    returns_for_state = returns[state]
                    returns_for_state.append(G)
                    returns[state] = returns_for_state
                else:
                    returns[state] = [G]
                Q[state] = (sum(returns[state]) /
                                   len(returns[state]))

    return Q


def main():
    game = Blackjack
    Q = mc_prediction(game, 1000, choose_rand, 0.5)
    print(Q)


def choose_rand(state: List[int]) -> int:
    if state[3] == 0:
        if state[2] > 19:
            return 0
        else:
            return 1
    else:
        if state[2] > 29:
            return 0
        else:
            return 1


if __name__ == '__main__':
    main()
