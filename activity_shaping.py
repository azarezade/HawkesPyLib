# __author__ = 'Ali_Zarezade'

import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

from numpy.linalg import inv
from scipy.linalg import logm, expm
from scipy.optimize import fsolve, brentq
import scipy.integrate as integrate
from event_generation import *
from activity_maximization import psi


def f(s, i, tf, alpha, w, d, ell, b):
    """
    f_i(s_i) + nu = 0
    sum s_i = c/b
    """
    n = alpha.shape[0]
    I = np.eye(n)
    g = psi(tf - s, alpha, w)[:, i].dot(d)
    psi_int = I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * (tf - s)) - expm((alpha - w * I) * tf))
    g_int = psi_int[:, i].dot(d)
    return g * (b * g_int - ell[i]) * 2 * d[i]


def maximize_shaping(b, c, d, ell, t0, tf, mu, alpha, w, tol=1e-1):
    """
    Solve the following optimization:
    """
    n = alpha.shape[0]
    t = np.zeros(n)

    I = np.eye(n)
    eta = (I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * tf) - I)).dot(mu)

    ub = max([f(t0, i, tf, alpha, w, d, ell-eta[i], b) for i in range(n)])
    lb = min([f(tf, i, tf, alpha, w, d, ell-eta[i], b) for i in range(n)])
    while abs(sum(t) * b - c) > tol:
        m = (ub + lb) / 2.  # m = -nu
        for i in range(n):
            if f(t0, i, tf, alpha, w, d, ell-eta[i], b) < m:
                t[i] = t0
            else:
                t[i] = brentq(lambda s: f(s, i, tf, alpha, w, d, ell-eta[i], b) - m, t0, tf)
                k = i
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(t_i) for t_i in t], f(t[k], k, tf, alpha, w, d, ell-eta[i], b) - m, sum(t) * b - c))
        if sum(t) * b > c:
            lb = m
        else:
            ub = m
    return t


def eval_shaping(t, u, d, ell, t0, tf, alpha, w):
    n = alpha.shape[0]
    integral = np.zeros(n)
    for i in range(n):
        integral[i] = u(t)[i] + integrate.quad(lambda s: u(s)[i] * psi(tf - s, alpha, w)[:, i].dot(d), t0, tf)[0]
    return np.linalg.norm(integral - ell) ** 2


def main():
    np.random.seed(0)
    t0 = 0
    tf = 100
    n = 8
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 1

    ell = 10 * np.ones(n)

    b = 100 * mu_max
    c = 10 * tf * mu_max
    d = np.ones(n)

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    t_opt = maximize_shaping(b, c, d, ell, t0, tf, mu, alpha, w)
    print(eval_shaping(tf, lambda s: [b * (s < t_opt[j]) for j in range(n)], d, ell, t0, tf, alpha, w))

    # tt = np.arange(t0, tf, 1)
    # yy = np.zeros(len(tt))
    # for i in range(n):
    #     for k in range(len(tt)):
    #         yy[k] = f(tt[k], i, tf, alpha, w, d, ell, b)
    #     plt.plot(tt, yy)
    # plt.show()


if __name__ == '__main__':
    main()

