import random
import math
import numpy as np
import pdb
import copy
import matplotlib.pyplot as plt

from common import *


Q_MC = np.load("QFromMonteCarlo.npy")
dealerLimits = np.array([[1, 4],
                         [4, 7],
                         [7, 10]])
ownLimits = np.array([[1, 6],
                      [4, 9],
                      [7, 12],
                      [10, 15],
                      [13, 18],
                      [16, 21]])
actionLimits = np.array([[0, 0],
                         [1, 1]])
numFeatures = dealerLimits.shape[0] * ownLimits.shape[0] * actionLimits.shape[0]

# precalculate features
allX = np.zeros([10, 21, 2, numFeatures])
for i in range(10):
    for j in range(21):
        for k in range(2):
            dealersCard = i+1
            ownSum = j+1
            action = k

            # find indices for intervals
            dealerIdxs = np.logical_and(dealersCard >= dealerLimits[:, 0],
                                        dealersCard <= dealerLimits[:, 1])
            ownIdxs = np.logical_and(ownSum >= ownLimits[:, 0],
                                     ownSum <= ownLimits[:, 1])
            actionIdxs =  np.logical_and(action >= actionLimits[:, 0],
                                         action <= actionLimits[:, 1])

            # combine indices
            dealer3D = np.zeros([dealerLimits.shape[0], ownLimits.shape[0], actionLimits.shape[0]])
            dealer3D[dealerIdxs, :, :] = 1
            own3D = np.zeros([dealerLimits.shape[0], ownLimits.shape[0], actionLimits.shape[0]])
            own3D[:, ownIdxs, :] = 1
            action3D = np.zeros([dealerLimits.shape[0], ownLimits.shape[0], actionLimits.shape[0]])
            action3D[:, :, actionIdxs] = 1
            features3D = np.logical_and(dealer3D, np.logical_and(own3D, action3D))

            # save to all features
            allX[i, j, k, :] = features3D.flatten().astype(float)

            
# returns feature vector from the precalculated array
def getX(state, action):
    return allX[state.dealersCard-1, state.ownSum-1, action].reshape((numFeatures, 1))


# returns action value function
def getQ(features, weights):
    return features.transpose() @ weights
    

# returns MSE of backward SARSA(lambda) with linear function approximation
def sarsaApprox(lambda_, printProgress=False):
    # init
    numEpizodes = 2000
    epsilon = 0.05
    alpha = 0.01
    W = np.random.rand(numFeatures, 1) # weights
    MSE = np.zeros(numEpizodes)

    # loop of epizodes
    for t in range(numEpizodes):
        # init eligibility traces
        E = np.zeros([numFeatures, 1]) # eligibility traces
        # draw starting cards and choose initial action
        state = State(newCard(True), newCard(True))
        action = 0
        X = getX(state, action) # features
        if printProgress:
            print('Starting with state/action', state.dealersCard, state.ownSum, action)

        # unfold trajectory, until terminal state is reached
        while not state.isTerminalState():
            # save state and action, so that I can look for future values
            currentX = copy.deepcopy(X)

            # do a step
            reward = step(state, action, printProgress)

            # from future state, decide what future action to take
            action = math.nan
            X_0 = getX(state, 0) # features for action 0
            X_1 = getX(state, 1) # features for action 0
            if not state.isTerminalState():
                if random.random() < epsilon:
                    # pick random action
                    action = random.randint(0, 1)
                    if printProgress:
                        print('Picking random action', action)
                else:
                    # pick the best action
                    action = np.argmax([getQ(X_0,W), getQ(X_1,W)])
                    if printProgress:
                        print('Picking best action', action)
            # save features according to the picked action
            X = X_0 if action == 0 else X_1
            if printProgress:
                print('Future state/action is', state.dealersCard, state.ownSum, action)

            # knowing current and future state/action, let's update W and E
            if not state.isTerminalState():
                delta = reward + getQ(X,W) - getQ(currentX,W)
            else:
                delta = reward - getQ(currentX,W)
            E += currentX
            if printProgress:
                print('Reward is',reward,', updating W and E.')
            W += alpha * np.multiply(delta, E)
            E *= lambda_

        # calculate MSE and save it
        Q = np.zeros([10, 21, 2])
        for i in range(10):
            for j in range(21):
                for k in range(2):
                    Q[i, j, k] = getQ(getX(State(i+1, j+1), k), W)
        MSE[t] = np.mean(np.square(Q - Q_MC), axis=(0, 1, 2))

    return MSE


# plots MSE as a function of lambda and episodes
def plotSarsaApprox():
    lambdaRange = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plotLambda = [True, False, False, False, False, False, False, False, False, False, True]
    mseVsLambda = [0] * len(lambdaRange)

    fig0, ax0 = plt.subplots()
    
    for i_lambda in range(len(lambdaRange)):
        mse = sarsaApprox(lambdaRange[i_lambda])
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
