import random
import math
import numpy as np
import pdb
import copy
import matplotlib.pyplot as plt

from common import *

Q_MC = np.load("QFromMonteCarlo.npy") # action value function from Monte Carlo learning

def sarsa(lambda_, printProgress=False):
    # init
    N0 = 100
    numEpizodes = 50000
    Q = np.zeros([10, 21, 2]) # action value function Q
    visitCount = np.zeros([10, 21, 2]) # visit count per state and action
    MSE = np.zeros(numEpizodes)

    # loop of epizodes
    for t in range(numEpizodes):
        # init eligibility traces
        E = np.zeros([10, 21, 2])
        # draw starting cards and choose initial action
        state = State(newCard(True), newCard(True))
        action = 0
        if printProgress:
            print('Starting with state/action', state.dealersCard, state.ownSum, action)

        # unfold trajectory, until terminal state is reached
        while not state.isTerminalState():
            # save count
            visitCount[state.dealersCard-1, state.ownSum-1, action] += 1
            if printProgress:
                print('Updating count at',state.dealersCard, state.ownSum, action,'to',visitCount[state.dealersCard-1, state.ownSum-1, action])
            
            # save state and action, so that I can look for future values
            currentState = copy.deepcopy(state)
            currentAction = copy.deepcopy(action)
            if printProgress:
                print('Current state/action is', currentState.dealersCard, currentState.ownSum, currentAction)

            # do a step
            reward = step(state, action, printProgress)

            # from future state, decide what future action to take
            action = math.nan
            if not state.isTerminalState():
                N_st = np.sum(visitCount[state.dealersCard-1, state.ownSum-1, :]) # number of state visits
                epsilon_t = N0 / (N0 + N_st)
                if printProgress:
                    print('Epsilon is',epsilon_t)
                if random.random() < epsilon_t:
                    # pick random action
                    action = random.randint(0, 1)
                    if printProgress:
                        print('Picking random action', action)
                else:
                    # pick the best action
                    action = np.argmax(Q[state.dealersCard-1, state.ownSum-1, :])
                    if printProgress:
                        print('Picking best action', action)
            if printProgress:
                print('Future state/action is', state.dealersCard, state.ownSum, action)

            # knowing current and future state/action, let's update Q and E
            if not state.isTerminalState():
                delta = reward + Q[state.dealersCard-1, state.ownSum-1, action] - Q[currentState.dealersCard-1, currentState.ownSum-1, currentAction]
            else:
                delta = reward - Q[currentState.dealersCard-1, currentState.ownSum-1, currentAction]
            E[currentState.dealersCard-1, currentState.ownSum-1, currentAction] += 1
            alpha = 1 / visitCount[currentState.dealersCard-1, currentState.ownSum-1, currentAction]
            if printProgress:
                print('Alpha is',alpha,', reward is',reward,', updating Q and E at all states/actions.')
            Q += alpha * np.multiply(delta, E)
            E *= lambda_

        # calculate MSE and save it
        MSE[t] = np.mean(np.square(Q - Q_MC), axis=(0, 1, 2))

    return MSE


def plotSarsa():
    lambdaRange = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plotLambda = [True, False, False, False, False, False, False, False, False, False, True]
    mseVsLambda = [0] * len(lambdaRange)

    fig0, ax0 = plt.subplots()
    
    for i_lambda in range(len(lambdaRange)):
        mse = sarsa(lambdaRange[i_lambda])
        mseVsLambda[i_lambda] = mse[-1]
        if plotLambda[i_lambda]:
            ax0.plot(range(len(mse)), mse, label='lambda = ' + str(lambdaRange[i_lambda]))

    ax0.set(xlabel='Episodes', ylabel='MSE')
    ax0.grid()
    ax0.legend()

    fig1, ax1 = plt.subplots()
    ax1.plot(lambdaRange, mseVsLambda)
    ax1.set(xlabel='lambda', ylabel='MSE')
    ax1.grid()
    
    plt.show()
        
        
