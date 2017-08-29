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
from activity_maximization import psi, psi_int, eval_weighted_activity


def f1(s, i, t0, tf, alpha, w, d, ell, b):
    """
    f_i(s_i) + nu = 0
    sum s_i = c/b
    """
    n = alpha.shape[0]
    I = np.eye(n)
    g = psi(tf - s, alpha, w)[:, i].dot(d)
    _psi_int = I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * (tf - s)) - expm((alpha - w * I) * tf))
    g_int = _psi_int[:, i].dot(d)
    return g * (b * g_int - ell[i]) * 2 * d[i]


def f2(s, i, t0, tf, alpha, w, d, ell, b):
    """
    f_i(s_i) + nu = 0
    sum s_i = c/b
    """
    n = alpha.shape[0]
    I = np.eye(n)
    g = psi_int(s, t0, tf, alpha, w)[:, i].dot(d)
    psi_int_int = I * s - alpha.dot(inv(alpha - w * I)).\
        dot(inv(alpha - w * I).dot(expm((alpha - w * I) * (tf - s)) + I * s))
    g_int = psi_int_int[:, i].dot(d)
    return g * (b * g_int - ell[i]) * 2 * d[i]


def maximize_shaping1(b, c, d, ell, t0, tf, mu, alpha, w, tol=1e-1):
    """
    Solve the following optimization:
    """
    n = alpha.shape[0]
    t = np.ones(n)

    # I = np.eye(n)
    # eta = (I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * tf) - I)).dot(mu)

    ub = max([f1(t0, i, t0, tf, alpha, w, d, ell, b) for i in range(n)])
    lb = min([f1(tf, i, t0, tf, alpha, w, d, ell, b) for i in range(n)])
    while abs(sum(t) * b - c) > tol:
        m = (ub + lb) / 2.  # m = -nu
        for i in range(n):
            if f1(t0, i, t0, tf, alpha, w, d, ell, b) < m:
                t[i] = t0
            elif f1(tf, i, t0, tf, alpha, w, d, ell, b) > m:
                t[i] = tf
            else:
                t[i] = brentq(lambda s: f1(s, i, t0, tf, alpha, w, d, ell, b) - m, t0, tf)
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(t_i) for t_i in t], f1(t[i], i, t0, tf, alpha, w, d, ell, b) - m, sum(t) * b - c))
        if sum(t) * b > c:
            lb = m
        else:
            ub = m
    return t


def maximize_shaping2(b, c, d, ell, t0, tf, mu, alpha, w, tol=1e-1):
    """
    Solve the following optimization:
    """
    n = alpha.shape[0]
    t = np.zeros(n)

    # I = np.eye(n)
    # eta = (I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * tf) - I)).dot(mu)
    # eta_int = np.zeros(n)
    # for i in range(n):
    #     eta_int[i] = integrate.quad(lambda s: mu[i] * psi_int(tf - s, alpha, w)[:, i].dot(d), t0, tf)[0]

    ub = max([f2(tf, i, t0, tf, alpha, w, d, ell, b) for i in range(n)])
    lb = min([f2(t0, i, t0, tf, alpha, w, d, ell, b) for i in range(n)])
    while abs(sum(t) * b - c) > tol:
        m = (ub + lb) / 2.  # m = -nu
        # if m > 0:
        for i in range(n):
            # if f2(t0, i, t0, tf, alpha, w, d, ell, b) < m:
            #     t[i] = t0
            # elif f2(tf, i, t0, tf, alpha, w, d, ell, b) > m:
            #     t[i] = tf
            # else:
            t[i] = brentq(lambda s: f2(s, i, t0, tf, alpha, w, d, ell, b) - m, t0, tf)
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(t_i) for t_i in t], f2(t[i], i, t0, tf, alpha, w, d, ell, b) - m, sum(t) * b - c))
        if sum(t) * b > c:
            ub = m
        else:
            lb = m
    return t


def eval_shaping1(t, u, d, ell, t0, tf, alpha, w):
    n = alpha.shape[0]
    integral = np.zeros(n)
    for i in range(n):
        integral[i] = u(t)[i] + integrate.quad(lambda s: u(s)[i] * psi(t - s, alpha, w)[:, i].dot(d), t0, tf)[0]
    obj = np.linalg.norm(integral - ell)
    print(integral)
    print(ell)
    print(obj)
    return obj


def eval_shaping2(u, d, ell, t0, tf, alpha, w):
    n = alpha.shape[0]
    integral = np.zeros(n)
    for i in range(n):
        integral[i] = integrate.quad(lambda s: u(s)[i] * psi_int(s, t0, tf, alpha, w)[:, i].dot(d), t0, tf)[0]
    obj= np.linalg.norm(integral - ell)
    print(integral)
    print(ell)
    print(obj)
    return obj


def main():
    # np.random.seed(100)
    t0 = 0
    tf = 100
    n = 8
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 1

    ell = np.array([2, 2, 2, 4, 4, 4, 6, 4])  # 3 * np.ones(n)
    # ell = 5 * np.random.rand(n)
    # ell = 0.3 * np.ones(n)

    b = 100 * mu_max
    c = 100 * tf * mu_max
    d = np.ones(n)

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    # t_opt = maximize_shaping1(b, c, d, ell, t0, tf, mu, alpha, w)
    # eval_shaping1(tf, lambda s: [b * (s < t_opt[j]) for j in range(n)], d, ell, t0, tf, alpha, w)

    t_opt = maximize_shaping2(b, c, d, ell, t0, tf, mu, alpha, w)
    eval_shaping2(lambda s: [b * (s < t_opt[j]) for j in range(n)], d, ell, t0, tf, alpha, w)

    # tt = np.arange(t0, tf, 1)
    # yy = np.zeros(len(tt))
    # for i in range(n):
    #     for k in range(len(tt)):
    #         yy[k] = f1(tt[k], i, t0, tf, alpha, w, d, ell, b)
    #     plt.plot(tt, yy)
    # plt.show()


if __name__ == '__main__':
    main()
