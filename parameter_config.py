__author__ = 'cyyen'

import numpy as np

# parameter
beta = 0.8
alpha = 1.5

# possible earning outcomes S
ys = [1, 0.05]
N_s = len(ys)
PI = np.array([[0.75, 0.25], [0.25, 0.75]])

# real interest rate
r = 0.04

# capital vector
a_min = - 0.525
a = np.arange(a_min, 2, 0.005)
N = len(a)

# legal record keeping parameter
rho = 0.9

# initial guess of q
q_0 = np.ones((N, N_s)) * (1/(1+r))
q_min = np.ones((N, N_s)) * 0.9
q_max = np.ones((N, N_s)) * 1