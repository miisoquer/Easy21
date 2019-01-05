import random
import math


# State-defining class
class State():
    def __init__(self, dealersCard, ownSum):
        self.dealersCard = dealersCard
        self.ownSum = ownSum
    def setToTerminalState(self):
        self.dealersCard = 0
        self.ownSum = 0
    def isTerminalState(self):
        return self.dealersCard == 0 and self.ownSum == 0


# Draw a card
def newCard(startingCard=False):
    value = random.randint(1, 10)
    if not startingCard:
        # starting card is always positive
        color = random.random()
        if color <= 1/3:
            value *= -1
    return value


# Game function, takes state and action, updates state and returns reward
def step(state, action, printProgress=False):
    # checks
    assert isinstance(state, State), print('State must be of type State')
    assert state.dealersCard >= 1 and state.dealersCard <= 10, print('Impossible value of dealer\'s card')
    assert state.ownSum >= 1 and state.ownSum <= 21, print('Impossible value of own sum')
    assert action == 0 or action == 1, print('Wrong action, use 0 or 1')

    reward = 0 

    if action == 0:
        # hit action, taking new card
        state.ownSum += newCard()
        if printProgress:
            print('Taking a new card, new sum is',state.ownSum)

        # checking if I busted it
        if state.ownSum < 1 or state.ownSum > 21:
            state.setToTerminalState()
            reward = -1
            if printProgress:
                print('Busted, reward is',reward)
    elif action == 1:
        # stick action, sticking with what I have
        if printProgress:
            print('Sticking to',state.ownSum,'that I have')

        # dealer drawing cards
        dealersSum = state.dealersCard
        while dealersSum >= 1 and dealersSum < 17:
            dealersSum += newCard()
            if printProgress:
                print('Dealer is taking a new card, his new sum is',dealersSum)

        # evaluating dealer's results
        if dealersSum < 1 or dealersSum > 21 or dealersSum < state.ownSum:
            reward = 1
        elif dealersSum == state.ownSum:
            reward = 0
        else:
            reward = -1
        
        if printProgress:
            print('Own sum is',state.ownSum,', dealers sum is',dealersSum,', reward is',reward)
        state.setToTerminalState()

    return reward
