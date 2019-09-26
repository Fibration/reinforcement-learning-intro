import random
from typing import List


class Blackjack:
    """
    Single player blackjack game. \n
    get_actions() gives you all the available actions. \n
    play(action: int) will make a move. \n
    get_state() returns the current state
    [
        dealer hand value,
        dealer ace count,
        player hand value,
        player ace count
    ]
    """
    def __init__(self):
        self.deck = [card for card in
                     ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                      'J', 'Q', 'K']
                     * 16]
        random.shuffle(self.deck)
        self.terminal = False

        # deal two cards to player, one to dealer
        self.player_hand = [self.deck.pop()]
        self.player_hand.append(self.deck.pop())
        self.dealer_hand = [self.deck.pop()]
        print(f'Player hand: {self.player_hand},',
              f'Dealer hand: {self.dealer_hand}')

    def get_actions(self) -> int:
        return [0, 1]

    def get_state(self) -> List[int]:
        return [*self.evaluate(self.dealer_hand),
                *self.evaluate(self.player_hand)]

    def evaluate(self, hand: List[str]) -> (int, int):
        def convert(card: str) -> int:
            if card == 'A':
                return 11  # assume A is 11 and discount as needed
            elif card in ['J', 'Q', 'K']:
                return 10
            else:
                return int(card)

        def count_aces(cards: List[str]) -> int:
            return sum([1 if card == 'A' else 0 for card in cards])

        values = [convert(card) for card in hand]
        return sum(values), count_aces(hand)  # hand value, count of aces

    def play(self, action: int) -> (float, List[int]):
        """
        Play one of the available moves, represented as an int. \n
        Returns a reward, state.
        """
        # catch invalid actions
        if action not in self.get_actions():
            raise ValueError('Not a valid action!')

        if action == 0:  # stick
            print('Player sticks.')
            return self.terminate(), self.get_state()
        elif action == 1:  # hit
            print('Player hits.')
            new_card = self.deck.pop()
            print(f'Card is {new_card}')
            self.player_hand.append(new_card)

            # check if player is now bust and the game ends
            new_state = self.get_state()
            if self.score(new_state[2:4]) > 21:
                return self.terminate(new_state), new_state
            else:
                return 0, new_state

    def score(self, state: List[int]) -> int:
        """
        Get best score from a hand. \n
        Input: [hand value, count of aces] \n
        Returns best score.
        """
        value, aces = state
        while value > 21 and aces > 0:
            aces -= 1
            value -= 10

        return value

    def terminate(self, passed_state: List[int] = None) -> int:
        self.terminal = True
        state = None
        if passed_state is None:
            state = self.get_state()
        else:
            state = passed_state
        dealer_value = state[0]
        player_value = self.score(state[2:4])

        # check if player is bust;
        if player_value > 21:
            print('Player goes bust!')
            return -1

        # otherwise, dealer draws until hand is >= 17
        while dealer_value < 17:
            self.dealer_hand.append(self.deck.pop())
            state = self.get_state()
            dealer_value = self.score(state[0:2])

        if dealer_value > 21:
            print('Dealer goes bust!')
            return 1
        elif dealer_value > player_value:
            print('Dealer wins!')
            return -1
        elif dealer_value < player_value:
            print('Player wins!')
            return 1
        else:
            print('It\' a draw!')
            return 0


if __name__ == '__main__':
    blackjack = Blackjack()
    print(blackjack.play(1))
    while blackjack.terminal is False:
        if blackjack.get_state()[2] < 19:
            print(blackjack.play(1))
        else:
            print(blackjack.play(0))
