__author__ = 'cyyen'

import numpy as np
import pickle
import pdb

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


class HW6(object):
    def __init__(self, is_debug=False):
        self.DEBUG = is_debug

        # parameter
        self.beta = 0.8
        self.alpha = 1.5

        # possible earning outcomes S
        self.ys = [1, 0.05]
        self.PI = np.array([[0.75, 0.25], [0.25, 0.75]])

        # real interest rate
        self.r = 0.04

        # capital vector
        a_min = - 0.525
        self.a = np.arange(a_min, 2, 0.05)# 0.005
        N = len(self.a)

        # legal record keeping parameter
        self.rho = 0.9

        # initial guess of q
        self.q_0 = np.ones((len(self.a), len(self.ys))) * (1/(1+self.r))

    def calculate_V_h1(self, v_init):
        ys, a, r, alpha, beta, rho, PI = self.ys, self.a, self.r, self.alpha, self.beta, self.rho, self.PI
        N_s = len(ys)
        N = len(a)

        # given V_0(s, a, h=0; I)
        v_h0 = v_init[0]

        # TARGET: calculate V(s, a, h=1; I)
        v_h1 = v_init[1]
        g_h1 = v_h1.copy().astype(int)

        # CONTAINER
        v_h1_new = v_h1.copy()

        # only positive a is feasible
        a_p = np.where(a >= 0, a, np.nan)

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
            if self.DEBUG:
                print('\t[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

        return v_h1, g_h1

    def calculate_V_h0_d0(self, v_init):
        ys, a, q_0, alpha, beta, PI = self.ys, self.a, self.q_0, self.alpha, self.beta, self.PI
        N_s = len(ys)
        N = len(a)

        # given V_0(s, a, h=0; I),
        # TARGET: calculate V_d0(s, a, h=0; I)
        v_h0 = v_init[0]
        g_h0 = v_h0.copy().astype(int)

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
            if self.DEBUG:
                print('\t[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

        return v_h0, g_h0

    def calculate_V_h0_d1(self, v_init):
        ys, a, alpha, beta, PI = self.ys, self.a, self.alpha, self.beta, self.PI
        N_s = len(ys)
        N = len(a)

        # given V(s, a, h=1; I)
        v_h1 = v_init[1]

        # TARGET: calculate V_d1(s, a, h=0; I)
        u = np.zeros((N, N_s))
        for s in range(N_s):
            u[:, s] = (ys[s] **(1-alpha)-1)/(1-alpha)

        idx_a_eq_0 = np.where(np.diff(np.sign(a)) != 0)[0][0]
        v_h0 = u + beta * v_h1[idx_a_eq_0, :].dot(PI.T)
        g_h0 = np.ones((N, N_s)).astype(int) * idx_a_eq_0

        return v_h0, g_h0

    def get_value_fn_iteration(self):
        ys, a = self.ys, self.a

        # define 3-d array v_init(s, a, h; I), the structure is as followed: v_init[h][a, s]
        N = len(a)
        N_s = len(ys)
        v_h0_init = np.zeros((N, N_s))
        v_h1_init = np.zeros((N, N_s))
        V_init = np.array([v_h0_init, v_h1_init])
        G = V_init.copy().astype(int)
        d = v_h0_init.copy()

        V_calculated = V_init.copy()

        pcntol=1
        iter_no = 0
        while pcntol > 10**(-3):
            # (1) given V_0(s, a, h=0; I), calculate V(s, a, h=1; I)
            V_calculated[1], G[1] = self.calculate_V_h1(V_init)

            #(2) given V(s, a, h=1; I), calculate V_1(s, a, h=0; I)
            V_h0_d0, G_h0_d0 = self.calculate_V_h0_d0(V_init)
            V_h0_d1, G_h0_d1 = self.calculate_V_h0_d1(V_calculated)
            V_calculated[0] = np.where(V_h0_d0 > V_h0_d1, V_h0_d0, V_h0_d1)
            G[0] = np.where(V_h0_d0 > V_h0_d1, G_h0_d0, G_h0_d1)
            d = np.where(V_h0_d0 > V_h0_d1, 0, 1)

            tol = abs(V_calculated - V_init).max()
            pcntol = tol/abs(V_calculated).min()
            V_init = V_calculated.copy()

            iter_no += 1
            if self.DEBUG:
                print('[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

        return V_init, G, d




hw6 = HW6(is_debug=False)

N = len(hw6.a)
N_s = len(hw6.ys)
v_h0_init = np.zeros((N, N_s))
v_h1_init = np.zeros((N, N_s))
V_init = np.array([v_h0_init, v_h1_init])
G = V_init.copy().astype(int)
d = v_h0_init.copy()

V_calculated = V_init.copy()
q_0 = hw6.q_0

# V_calculated[1], G[1] = hw6.calculate_V_h1(V_init)
# V_h0_d0, G_h0_d0 = hw6.calculate_V_h0_d0(V_init)
# V_h0_d1, G_h0_d1 = hw6.calculate_V_h0_d1(V_calculated)

q_min = np.ones((N, N_s)) * 0
q_max = np.ones((N, N_s)) * 1#?

pcntol = 1
iter_no = 0
while pcntol > 10**(-2):
    V, G, d = hw6.get_value_fn_iteration()

    q_1 = q_0.copy()
    for i in range(N):
        for s in range(N_s):
            delta = hw6.PI[s, 0]*d[i, 0] + hw6.PI[s, 1]*d[i, 1]
            q_1[i, s] = (1 - delta)/(1+hw6.r)

    # bisection method
    q_min = np.where(q_1 >= q_0, q_0, q_min)
    print q_min
    print '~~~'

    q_max = np.where(q_1 < q_0, q_0, q_max)
    print q_max
    print '~~~'

    q_0 = (q_min + q_max)/2
    print q_0
    print '~~~~~~~'

    tol = abs(q_1 - q_0).max()
    pcntol = tol#/np.mean(abs(q_1))


    iter_no += 1
    # if DEBUG:
    print('[V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))


# store computed results
with open("results/g.p", "wb") as f:
    pickle.dump(G, f)

with open("results/v.p", "wb") as f:
    pickle.dump(V, f)

with open(open("results/d.p", "wb")) as f:
    pickle.dump(d, f)

with open("results/q.p", "wb") as f:
    pickle.dump(q_0, f)