#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/02/02 23:35

@author: Tei Koten
"""

import numpy as np

def HMM(A, B, PI, obs=[0, 1, 0]):
    """
    Hidden Markov Model Forward Algorithm
    EXAMPLE:
    ------------------------------
    A = np.array([
        [0.5,0.2,0.3],
        [0.3,0.5,0.2],
        [0.2,0.3,0.5]])
    B = np.array([
        [0.5,0.5],
        [0.4,0.6],
        [0.7,0.3]])
    PI = np.array([0.2,0.4,0.4])
    model = lambda(A,B,PI)
    ------------------------------
    :param A: status transition matrix
    :param B: observation prob matrix
    :param PI: initial state prob
    :param obs: observation sequence
    :return: prob(obs|lambda)
    """
    for i, j in zip(range(3), obs):
        if i == 0:
            alpha = PI * B[:, j]
        else:
            alpha = np.dot(alpha.reshape(1, -1), A) * B[:, j]
        print('state:' + str(i), alpha)
    res = np.sum(alpha)
    print('final res:', res)
    return res

if __name__ == '__main__':

    A = np.array([
        [0.5,0.2,0.3],
        [0.3,0.5,0.2],
        [0.2,0.3,0.5]])
    B = np.array([
        [0.5,0.5],
        [0.4,0.6],
        [0.7,0.3]])
    PI = np.array([0.2,0.4,0.4])

    res = HMM(A,B,PI)



