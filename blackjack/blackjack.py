'''
Blackjack
'''
import random

CARDS = '23456789JQKA'
SUITS = 'HCSD'


class Player():
    def __init__(self):
        self.cards = dict()
        self.dealer_hand = None
        self.value = 0

    def receive_card(self, card):
        self.cards[card] = self.cards.get(card, 0) + 1

    def set_dealer_hand(self, card):
        self.dealer_hand = card
        print('Dealer hand: {}'.format(card))

    def play(self, inplay):
        state = []
        value = 0
        for key in CARDS:
            state.append(self.cards.get(key, 0))
            value += self.get_card_value(key) * self.cards.get(key, 0)
        state.append(value)
        state.append(self.dealer_hand)
        state.extend(inplay)
        self.value = value
        print(self.cards)
        print(state)
        print(value)
        if value < 17:
            return 'HIT'
        else:
            return 'STAY'

    def get_card_value(self, card):
        if card in 'JQK':
            return 10
        elif card in '23456789':
            return int(card)
        elif card == 'A':
            return 11

    def get_value(self):
        return self.value


class Blackjack:
    def __init__(self, number_players, num_decks=2):
        self.players = [Player() for i in range(number_players)]
        self.dealer = []
        deck = [i for i in CARDS for j in SUITS for k in range(num_decks)]
        random.shuffle(deck)
        self.deck = iter(deck)
        self.inplay = dict()

    def deal(self):
        deck = self.deck
        for player in self.players:
            # deal two cards to the player
            cards = [next(deck) for i in range(2)]
            for card in cards:
                player.receive_card(card)
                self.inplay[card] = self.inplay.get(card, 0) + 1

        # deal one closed card to the dealer and then deal the other to face up
        self.dealer.append(next(deck))
        dealer_card = next(deck)
        self.dealer.append(dealer_card)
        self.inplay[dealer_card] = self.inplay.get(dealer_card, 0) + 1
        for player in self.players:
            player.set_dealer_hand(dealer_card)

        # update the deck
        self.deck = deck

    def play(self):
        # dela cards
        self.deal()
        deck = self.deck

        for player in self.players:
            move = 'NOTHING'
            while move != 'STAY':
                move = player.play(self.inplay)
                if move == 'HIT':
                    card = next(deck)
                    player.receive_card(card)
                    self.inplay[card] = self.inplay.get(card, 0) + 1
                    self.deck = deck

        for player in self.players:
            value = player.get_value()
            if value > 21:
                print("BUST")
            elif value > 19:
                print("GAME WON")
            else:
                print("GAME LOST")


def main():
    blackjack = Blackjack(1)
    blackjack.play()


if __name__ == "__main__":
    main()
