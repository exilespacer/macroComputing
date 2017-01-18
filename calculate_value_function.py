__author__ = 'cyyen'

import numpy as np
import pickle
import pdb


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
        self.a = np.arange(a_min, -a_min, 0.01)# 0.005

        # legal record keeping parameter
        self.rho = 0.9

        # initial guess of q
        self.q_0 = np.ones((len(self.a), len(self.ys))) * 0.98 #* (1/(1+self.r))

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
        # a_p = np.where(a >= 0, a, np.nan)

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
                v = (u + beta * (rho * v_h1 + (1-rho) * v_h0).dot(PI.T))
                v[self.a < 0, :] = -np.inf
                v_h1_new[i, :] = v.max(axis=0)
                g_h1[i, :] = np.argmax(v, axis=0)

            tol = abs(v_h1_new - v_h1).max()
            pcntol = tol/abs(v_h1_new).min()
            v_h1 = v_h1_new.copy()

            iter_no += 1
            if self.DEBUG:
                print('\t\t[V_h1][V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

        return v_h1, g_h1

    def calculate_V_h0_d0(self, v_init, q_0):
        ys, a, alpha, beta, PI = self.ys, self.a, self.alpha, self.beta, self.PI
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
                v = u + beta * v_h0.dot(PI.T)
                v[self.a >= 0, :] = -np.inf
                v_h0_new[i, :] = (v).max(axis=0)
                g_h0[i, :] = np.argmax(v, axis=0)

            tol = abs(v_h0_new - v_h0).max()
            pcntol = tol/abs(v_h0_new).min()
            v_h0 = v_h0_new.copy()

            iter_no += 1
            if self.DEBUG:
                print('\t\t[V_h0_d0][V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

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

    def calculate_V_by_value_fn_iteration(self, q_0):
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
            # (1) calculate V(s, a, h=1; I)
            V_calculated[1], G[1] = self.calculate_V_h1(V_init)

            #(2) calculate V_1(s, a, h=0; I)
            V_h0_d0, G_h0_d0 = self.calculate_V_h0_d0(V_init, q_0)

            V_h0_d1, G_h0_d1 = self.calculate_V_h0_d1(V_init)
            # pdb.set_trace()

            V_calculated[0] = np.where(V_h0_d0 > V_h0_d1, V_h0_d0, V_h0_d1)
            G[0] = np.where(V_h0_d0 > V_h0_d1, G_h0_d0, G_h0_d1)
            d = np.where(V_h0_d0 > V_h0_d1, 0, 1)
            d[self.a > 0, :] = 0

            tol = abs(V_calculated - V_init).max()
            pcntol = tol/abs(V_calculated).min()
            V_init = V_calculated.copy()

            iter_no += 1
            if self.DEBUG:
                print('\t[V/G/d][V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

        return V_init, G, d

    def calculate_state_variables_in_separating_equilibrium(self):
        q_0 = self.q_0.copy()
        N, N_s = q_0.shape

        q_min = np.zeros((N, N_s))
        q_max = np.ones((N, N_s))

        pcntol = 1
        iter_no = 0
        while pcntol > 10**(-3):
            V, G, d = self.calculate_V_by_value_fn_iteration(q_0)

            [[p00, p01], [p10, p11]] = self.PI
            delta = np.array([d[:, 0] * p00 + d[:, 1] * p01, d[:, 0] * p10 + d[:, 1] * p11]).T
            q_1 = (1 - delta)/(1+self.r)

            pcntol = abs(q_1 - q_0).max()

            # bisection method
            q_min = np.where(q_1 >= q_0, q_0, q_min)
            q_max = np.where(q_1 <= q_0, q_0, q_max)
            q_0 = ((q_min + q_max)/2).copy()

            iter_no += 1
            print('[separating equilibrium][V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))

        return V, G, d, q_0

    def calculate_asyn_distribution(self, V, G, d):
        Mu_0 = np.ones(V.shape)/np.prod(V.shape)
        mu_h0 = Mu_0[0].copy()
        mu_h1 = Mu_0[1].copy()

        g_h0 = G[0]
        g_h1 = G[1]

        pcntol = 1
        iter_no = 0
        while pcntol > 10**(-3):
            Mu_1 = Mu_0.copy()

            # idx_a1: the index of next period a value
            # idx_a0: the index of current period a value
            N, N_s = g_h0.shape
            for idx_a1 in range(N):
                # find the index of "a whose next period value of a' "
                idx_a0_h0_s0 = (g_h0[:, 0] == idx_a1)
                idx_a0_h0_s1 = (g_h0[:, 1] == idx_a1)
                idx_a0_h1_s0 = (g_h1[:, 0] == idx_a1)
                idx_a0_h1_s1 = (g_h1[:, 1] == idx_a1)

                # find the index of "default d" when h=0
                idx_default_h0_s0 = d[:, 0]
                idx_default_h0_s1 = d[:, 1]

                # probability of transition matrix, where p_xy means the probability from state x to state y
                [[p00, p01], [p10, p11]] = self.PI

                # mu(s'=0 | h=0) = mu(no flag, no default|s'=0) * prob(s'=0) + (1-rho) * mu(has flag|s'=0) * prob(s'=0)
                #       where
                #           mu(no flag, no default|s'=0)* prob(s'=0)
                #               =   mu(no flag, no default|s'=0, s= 0)* prob(s'=0, s=0) +
                #                   mu(no flag, no default|s'=0, s= 1)* prob(s'=0, s=1)
                #
                #           mu(has flag|s'=0) * prob(s'=0)
                #               =   mu(has flag|s'=0, s=0) * prob(s'=0, s=0) +
                #                   mu(has flag|s'=0, s=1) * prob(s'=0, s=1)

                mu_h0[idx_a1, 0] = (mu_h0[np.logical_and(idx_a0_h0_s0, 1- idx_default_h0_s0), 0].sum() * p00 + \
                                    mu_h0[np.logical_and(idx_a0_h0_s1, 1- idx_default_h0_s1), 1].sum() * p10) + \
                                   (1-self.rho) * (mu_h1[idx_a0_h1_s0, 0].sum() * p00 +
                                                  mu_h1[idx_a0_h1_s1, 1].sum() * p10)

                # mu(s'=1 | h=0) = mu(no flag, no default|s'=1) * prob(s'=1) + (1-rho) * mu(has flag|s'=1) * prob(s'=1)
                mu_h0[idx_a1, 1] = (mu_h0[np.logical_and(idx_a0_h0_s0, 1- idx_default_h0_s0), 0].sum() * p01 + \
                                    mu_h0[np.logical_and(idx_a0_h0_s1, 1- idx_default_h0_s1), 1].sum() * p11) + \
                                   (1-self.rho) * (mu_h1[idx_a0_h1_s0, 0].sum() * p01 +
                                                  mu_h1[idx_a0_h1_s1, 1].sum() * p11)

                # mu(s'= 0 | h=1) = mu(no flag, default|s'=0) * prob(s'=0) + (rho) * mu(has flag|s'=0) * prob(s'=0)
                mu_h1[idx_a1, 0] = (mu_h0[np.logical_and(idx_a0_h0_s0, idx_default_h0_s0), 0].sum() * p00 + \
                                    mu_h0[np.logical_and(idx_a0_h0_s1, idx_default_h0_s1), 1].sum() * p10) + \
                                   (self.rho) * (mu_h1[idx_a0_h1_s0, 0].sum() * p00 +
                                                  mu_h1[idx_a0_h1_s1, 1].sum() * p10)

                # mu(s'= 1 | h=1) = mu(no flag, default|s'=1) * prob(s'=1) + (rho) * mu(has flag|s'=1) * prob(s'=1)
                mu_h1[idx_a1, 1] = (mu_h0[np.logical_and(idx_a0_h0_s0, idx_default_h0_s0), 0].sum() * p01 + \
                                    mu_h0[np.logical_and(idx_a0_h0_s1, idx_default_h0_s1), 1].sum() * p11) + \
                                   (self.rho) * (mu_h1[idx_a0_h1_s0, 0].sum() * p01 +
                                                  mu_h1[idx_a0_h1_s1, 1].sum() * p11)

            Mu_1[0] = mu_h0
            Mu_1[1] = mu_h1

            pcntol = abs(Mu_1 - Mu_0).max()
            Mu_0 = Mu_1.copy()
            iter_no += 1
            if self.DEBUG:
                print('\t\t[Asyn. distribution]#: %s ; pcntol: %s' % (iter_no, pcntol))
        return Mu_0

    def calculate_state_variables_in_pooling_equilibrium(self):
        q_0 = self.q_0.copy()

        iter_no = 0
        pcntol = 1
        while pcntol > 10**(-3):
            V, G, d = hw6.calculate_V_by_value_fn_iteration(q_0)
            Mu = hw6.calculate_asyn_distribution(V, G, d)

            [[p00, p01], [p10, p11]] = hw6.PI

            mu_h0 = Mu[0]

            g_h0 = G[0]
            idx_a1_h0_s0 = g_h0[:, 0]
            idx_a1_h0_s1 = g_h0[:, 1]

            a1_h0_s0 = hw6.a[idx_a1_h0_s0]
            a1_h0_s1 = hw6.a[idx_a1_h0_s1]

            L = (a1_h0_s0 * mu_h0[:, 0])[a1_h0_s0 < 0].sum() + (a1_h0_s1 * mu_h0[:, 1])[a1_h0_s1 < 0].sum()

            d_next = np.array([d[idx_a1_h0_s0, 0] * p00 + d[idx_a1_h0_s0, 1] * p01,
                               d[idx_a1_h0_s1, 0] * p10 + d[idx_a1_h0_s1, 1] * p11]).T

            D = (a1_h0_s0 * d_next[:, 0] * mu_h0[:, 0])[a1_h0_s0 < 0].sum() + (a1_h0_s1 * d_next[:, 1] * mu_h0[:, 1])[a1_h0_s1 < 0].sum()

            q_1 = (1 - D/L)/(1+hw6.r)

            pcntol = abs(q_1 - q_0).max()
            q_0 = q_0.copy() - pcntol * 10**(-2) * 5

            iter_no += 1
            print('[pooling equilibrium][V.F.I.] #: %s ; pcntol: %s' % (iter_no, pcntol))
        return V, G, d, Mu, q_0



hw6 = HW6(is_debug=False)

# separating equilibrium
N, N_s = hw6.q_0.shape
V, G, d, q = hw6.calculate_state_variables_in_separating_equilibrium()
Mu = hw6.calculate_asyn_distribution(V, G, d)
print Mu

with open("results/separating/g.p", "wb") as f:
    pickle.dump(G, f)

with open("results/separating/v.p", "wb") as f:
    pickle.dump(V, f)

with open("results/separating/d.p", "wb") as f:
    pickle.dump(d, f)

with open("results/separating/q.p", "wb") as f:
    pickle.dump(q, f)

with open("results/separating/Mu.p", "wb") as f:
    pickle.dump(Mu, f)



# pooling equilibrium
V, G, d, Mu, q = hw6.calculate_state_variables_in_pooling_equilibrium()

# store computed results
with open("results/pooling/g.p", "wb") as f:
    pickle.dump(G, f)

with open("results/pooling/v.p", "wb") as f:
    pickle.dump(V, f)

with open("results/pooling/d.p", "wb") as f:
    pickle.dump(d, f)

with open("results/pooling/q.p", "wb") as f:
    pickle.dump(q, f)

with open("results/pooling/Mu.p", "wb") as f:
    pickle.dump(Mu, f)



