import random
from typing import List


class Blackjack:
    def __init__(self):
        self.deck = [card for card in
                     ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                      'J', 'Q', 'K']
                     * 16]
        random.shuffle(self.deck)
        self.player_hand = [self.deck.pop()]
        self.player_hand.append(self.deck.pop())
        self.dealer_hand = [self.deck.pop()]
        self.terminal = False
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
                return 11
            elif card in ['J', 'Q', 'K']:
                return 10
            else:
                return int(card)

        def count_aces(cards: List[str]) -> int:
            return sum([1 if card == 'A' else 0 for card in cards])

        values = [convert(card) for card in hand]
        return sum(values), count_aces(hand)

    def play(self, action: int) -> (float, List[int]):
        if action not in self.get_actions():
            raise ValueError('Not a valid action!')

        if action == 0:
            return self.terminate(), self.get_state()
        elif action == 1:
            new_card = self.deck.pop()
            self.player_hand.append(new_card)
            if self.score(self.get_state()[2:4]) > 21:
                return self.terminate(), self.get_state()
            else:
                return 0, self.get_state()

    def score(self, state: List[int]) -> int:
        value, aces = state
        while value > 21 and aces > 0:
            aces -= 1
            value -= 10

        return value

    def terminate(self) -> int:
        self.terminal = True
        state = self.get_state()
        dealer_value = state[0]
        player_value = self.score(state[2:4])
        if player_value > 21:
            print('Player goes bust!')
            return -1

        while dealer_value < 17:
            self.dealer_hand.append(self.deck.pop())
            state = self.get_state()
            dealer_value = self.score(state[0:2])

        print(dealer_value)
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
    print(blackjack.play(0))
