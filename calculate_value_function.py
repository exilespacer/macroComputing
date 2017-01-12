__author__ = 'cyyen'

import numpy as np
import pickle
import pdb
DEBUG = False

from parameter_config import *

#############################################################
# TARGET: CALCULATE VALUE FUNCTION V(s, a, h |I)
# PROCEDURE:
#   (1) given V_0(s, a, h=0; I), calculate V(s, a, h=1; I)
#   (2) given V(s, a, h=1; I), calculate V_1(s, a, h=0; I)
#           (a) given V_0(s, a, h=0; I), calculate V_d0(s, a, h=0; I)
#           (b) given V(s, a, h=1; I), calculate V_d1(s, a, h=0; I)
#           (c) calculate V_1(s, a, h=0; I) = max{V_d0, V_d1}
#   (3) Repeat (1)~(2) if |V_1(s, a, h=0; I) - V_0(s, a, h=0; I) - | > epsilon
#############################################################
def calculate_V_h1(v_init, ys, a, r, alpha, beta, rho, PI):
    N_s = len(ys)
    N = len(a)

    # given V_0(s, a, h=0; I)
    v_h0 = v_init[0]

    # TARGET: calculate V(s, a, h=1; I)
    v_h1 = v_init[1]
    g_h1 = v_h1.astype(int)

    # CONTAINER
    v_h1_new = v_h1.copy()

    # (1-1) calcualte utility function in advance, the structure is as followed: us[s][a, a']
    us = []
    for s in range(N_s):
        us_tmp = np.zeros((N, N))
        for i in range(N):
            c = ys[s] + a[i] - (1/(1+r)) * a
            c[c <= 0] = 0
            us_tmp[i, :] = (c**(1-alpha)-1)/(1-alpha)
        us_tmp[np.where(np.isinf(us_tmp))] = -10**(9)
        us.append(us_tmp.copy())

    # (1-2) calculate value function through Value function iteration
    pcntol=1
    iter_no = 0
    while pcntol > 10**(-3):
        u = np.zeros((N, N_s))
        for i in range(N):
            for s in range(N_s):
                u[:, s] = us[s][i, :]
            v_h1_new[i, :] = (u + beta * (rho * v_h1 + (1-rho) * v_h0).dot(PI.T)).max(axis=0)
            g_h1[i, :] = np.argmax(u + beta * (rho * v_h1 + (1-rho) * v_h0).dot(PI.T), axis=0)

        tol = abs(v_h1_new - v_h1).max()
        pcntol = tol/abs(v_h1_new).min()
        v_h1 = v_h1_new.copy()

        iter_no += 1
        if DEBUG:
            print('\t[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

    return v_h1, g_h1


def calculate_V_h0_d0(v_init, ys, a, q_0, alpha, beta, PI):
    N_s = len(ys)
    N = len(a)

    # given V_0(s, a, h=0; I),
    # TARGET: calculate V_d0(s, a, h=0; I)
    v_h0 = v_init[0]
    g_h0 = v_h0.astype(int)

    #CONTAINER
    v_h0_new = v_h0.copy()

    # (1-1) calcualte utility function in advance, the structure is as followed: us[s][a, a']
    us = []
    for s in range(N_s):
        us_tmp = np.zeros((N, N))
        for i in range(N):
            c = ys[s] + a[i] - q_0[:, s] * a
            c[c <= 0] = 0
            us_tmp[i, :] = (c**(1-alpha)-1)/(1-alpha)
        us_tmp[np.where(np.isinf(us_tmp))] = -10**(9)
        us.append(us_tmp.copy())


    # (1-2) calculate value function through Value function iteration
    pcntol = 1
    iter_no = 0
    while pcntol > 10**(-3):
        u = np.zeros((N, N_s))
        for i in range(N):
            for s in range(N_s):
                u[:, s] = us[s][i, :]
            v_h0_new[i, :] = (u + beta * v_h0.dot(PI.T)).max(axis=0)
            g_h0[i, :] = np.argmax(u + beta * v_h0.dot(PI.T), axis=0)

        tol = abs(v_h0_new - v_h0).max()
        pcntol = tol/abs(v_h0_new).min()
        v_h0 = v_h0_new.copy()

        iter_no += 1
        if DEBUG:
            print('\t[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

    return v_h0, g_h0


def calculate_V_h0_d1(v_init, ys, a, alpha, beta, PI):
    N_s = len(ys)
    N = len(a)

    # given V(s, a, h=1; I)
    v_h1 = v_init[1]

    # TARGET: calculate V_d1(s, a, h=0; I)
    u = np.zeros((N, N_s))
    for s in range(N_s):
        u[:, s] = (ys[s] **(1-alpha)-1)/(1-alpha)

    idx_a_eq_0 = np.where(abs(a) < 10**(-10))[0][0]
    v_h0 = u + beta * v_h1[idx_a_eq_0, :].dot(PI.T)
    g_h0 = np.ones((N, N_s)).astype(int) * idx_a_eq_0

    return v_h0, g_h0


def get_value_fn_iteration(ys, a, r, q_0, alpha, beta, rho, PI):
    # define 3-d array v_init(s, a, h; I), the structure is as followed: v_init[h][a, s]
    v_h0_init = np.zeros((N, N_s))
    v_h1_init = np.zeros((N, N_s))
    V_init = np.array([v_h0_init, v_h1_init])
    G = V_init.copy().astype(int)
    d = v_h0_init.copy()

    V_calculated = V_init.copy()

    pcntol=1
    iter_no = 0
    while pcntol > 10**(-3):
        ########################################################
        # (1) given V_0(s, a, h=0; I), calculate V(s, a, h=1; I)
        V_calculated[1], G[1] = calculate_V_h1(V_init, ys, a, r, alpha, beta, rho, PI)

        ########################################################
        #(2) given V(s, a, h=1; I), calculate V_1(s, a, h=0; I)
        #(a) given V_0(s, a, h=0; I), calculate V_d0(s, a, h=0; I)
        V_h0_d0, G_h0_d0 = calculate_V_h0_d0(V_init, ys, a, q_0, alpha, beta, PI)

        #(b) given V(s, a, h=1; I), calculate V_d1(s, a, h=0; I)
        V_h0_d1, G_h0_d1 = calculate_V_h0_d1(V_calculated, ys, a, alpha, beta, PI)

        #(c) calculate V_1(s, a, h=0; I) = max{V_d0, V_d1}
        V_calculated[0] = np.where(V_h0_d0 > V_h0_d1, V_h0_d0, V_h0_d1)
        G[0] = np.where(V_h0_d0 > V_h0_d1, G_h0_d0, G_h0_d1)
        d = np.where(V_h0_d0 > V_h0_d1, 0, 1)

        ########################################################
        #(3) Repeat (1)~(2) if |V_1(s, a, h=0; I) - V_0(s, a, h=0; I) - | > epsilon
        tol = abs(V_calculated - V_init).max()
        pcntol = tol/abs(V_calculated).min()
        V_init = V_calculated.copy()

        iter_no += 1
        if DEBUG:
            print('[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

    return V_init, G, d


pcntol=1
iter_no = 0
while pcntol < 10**(-3):
    V, G, d = get_value_fn_iteration(ys, a, r, q_0, alpha, beta, rho, PI)

    q_1 = q_0.copy()
    for i in range(N):
        for s in range(N_s):
            delta = PI[s, 0]*d[i, 0] + PI[s, 1]*d[i, 1]
            q_1[i, s] = (1 - delta)/(1+r)


    tol = abs(q_1 - q_0).max()
    pcntol = tol/abs(q_1).min()
    q_0 = q_1.copy()

    iter_no += 1
    # if DEBUG:
    print('[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))