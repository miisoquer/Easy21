import random
import math
import numpy as np
import pdb
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common import *

def monteCarlo(printProgress=False):
    # init
    N0 = 100
    Q = np.zeros([10, 21, 2]) # action value function
    visitCount = np.zeros([10, 21, 2]) # visit count per state and action

    # loop of epizodes
    for t in range(100000000):
        # draw starting cards
        state = State(newCard(True), newCard(True))
        if printProgress:
            print('Starting with state',state.dealersCard,state.ownSum)
        # init reward
        reward = math.nan
        # init visited states/actions in current episode
        trajectory = []

        # unfold trajectory, until terminal state is reached
        while not state.isTerminalState():
            # decide what action to take
            N_st = np.sum(visitCount[state.dealersCard-1, state.ownSum-1, :]) # number of state visits
            epsilon_t = N0 / (N0 + N_st)
            if printProgress:
                print('Epsilon is',epsilon_t)
            action = math.nan
            if random.random() < epsilon_t:
                # pick random action
                action = random.randint(0, 1)
                if printProgress:
                    print('Picking random action',action)
            else:
                # pick the best action
                action = np.argmax(Q[state.dealersCard-1, state.ownSum-1, :])
                if printProgress:
                    print('Picking best action',action)
            
            # save state and action, to know where to update Q
            # (I wouldn't need to save the action, cause only the last saved action is 1)
            trajectory.append((copy.deepcopy(state), copy.deepcopy(action)))
        
            # save count
            visitCount[state.dealersCard-1, state.ownSum-1, action] += 1
            if printProgress:
                print('Updating count at',state.dealersCard, state.ownSum, action,'to',visitCount[state.dealersCard-1, state.ownSum-1, action])
        
            # do a step
            reward = step(state, action, printProgress)
        
        # use reward to update Q
        for i_tr in range(len(trajectory)):
            # recover array indices
            idx0 = trajectory[i_tr][0].dealersCard - 1
            idx1 = trajectory[i_tr][0].ownSum - 1
            idx2 = trajectory[i_tr][1]
            # calculate alpha
            alpha_t = 1/visitCount[idx0, idx1, idx2]
            # update Q
            Q[idx0, idx1, idx2] += alpha_t*(reward - Q[idx0, idx1, idx2])
            if printProgress:
                print('Alpha_t is',alpha_t,', updating Q at',idx0+1, idx1+1, idx2,'to',Q[idx0, idx1, idx2])

    # save Q to file
    np.save('QFromMonteCarlo', Q, allow_pickle=False)
    # plot Q
    X = np.arange(1, 11, 1)
    Y = np.arange(1, 22, 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.transpose(np.amax(Q, axis=2))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 21)
    plt.yticks(np.arange(1, 22, 2))
    plt.xlabel('Dealer\'s visible card')
    plt.ylabel('Own sum')
    plt.show()
