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


def f(s, i, t0, tf, alpha, w, d, ell, b):
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


def f_int(s, i, t0, tf, alpha, w, d, ell, b):
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


def maximize_shaping(b, c, d, ell, t0, tf, mu, alpha, w, tol=1e-1):
    """
    Solve the following optimization:
    """
    n = alpha.shape[0]
    t = tf * np.ones(n)

    # I = np.eye(n)
    # eta = (I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * tf) - I)).dot(mu)

    ub = min([f(t0, i, t0, tf, alpha, w, d, ell, b) for i in range(n)])
    lb = max([f(tf, i, t0, tf, alpha, w, d, ell, b) for i in range(n)])
    while sum(t) * b - c > tol:
        m = (ub + lb) / 2.  # m = -nu
        for i in range(n):
            # if f(t0, i, t0, tf, alpha, w, d, ell, b) < m:
            #     t[i] = t0
            # elif f(tf, i, t0, tf, alpha, w, d, ell, b) > m:
            #     t[i] = tf
            # else:
            t[i] = brentq(lambda s: f(s, i, t0, tf, alpha, w, d, ell, b) - m, t0, tf)
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(t_i) for t_i in t], f(t[i], i, t0, tf, alpha, w, d, ell, b) - m, sum(t) * b - c))
        if sum(t) * b > c:
            lb = m
        else:
            ub = m
    return t


def maximize_shaping_int(b, c, d, ell, t0, tf, mu, alpha, w, tol=1e-1):
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

    tmp = np.concatenate(([f_int(tf, i, t0, tf, alpha, w, d, ell, b) for i in range(n)],
                          [f_int(t0, i, t0, tf, alpha, w, d, ell, b) for i in range(n)]))
    ub = max(tmp)
    lb = min(tmp)
    while abs(sum(t) * b - c) > tol:
        m = (ub + lb) / 2.  # m = -nu
        # if m > 0:
        for i in range(n):
            if f_int(t0, i, t0, tf, alpha, w, d, ell, b) > m:
                t[i] = t0
            else:
                t[i] = brentq(lambda s: f_int(s, i, t0, tf, alpha, w, d, ell, b) - m, t0, tf)
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(t_i) for t_i in t], f_int(t[i], i, t0, tf, alpha, w, d, ell, b) - m, sum(t) * b - c))
        if sum(t) * b > c:
            ub = m
        else:
            lb = m
    return t


def eval_shaping(s, b, d, ell, tf, alpha, w):
    n = alpha.shape[0]
    I = np.eye(n)
    Aw = alpha - w * I

    integral = np.zeros(n)
    for i in range(n):
        u_psi_int = (alpha.dot(inv(Aw)).dot(expm(Aw * tf) - expm(Aw * (tf - s[i]))))[:, i] + float(s[i] == tf) * I[:, i]
        integral[i] = b * u_psi_int.dot(d)

    obj = np.linalg.norm(integral - ell)
    print(integral)
    print(ell)
    print(obj)
    return obj


def eval_shaping_int(s, b, d, ell, tf, alpha, w):
    n = alpha.shape[0]
    I = np.eye(n)
    Aw = alpha - w * I
    Awi = inv(Aw)
    Awi2 = Awi.dot(Awi)

    integral = np.zeros(n)
    for i in range(n):
        u_psi = (I * s[i] + alpha.dot(Awi2).dot(expm(Aw * (tf - s[i])) - expm(Aw * tf)) - s[i] * alpha.dot(Awi))[:, i]
        integral[i] = b * u_psi.dot(d)

    obj = np.linalg.norm(integral - ell)
    print(np.concatenate((integral.reshape(1,n), ell.reshape(1,n)), axis=0))
    # print(ell)
    print(obj)
    return obj


def main():
    # np.random.seed(10)
    t0 = 0
    tf = 100
    n = 16
    sparsity = 0.1
    mu_max = 0.01
    alpha_max = 0.1
    w = 1

    # ell = np.array([2, 2, 2, 4, 4, 6, 6, 6])  # 3 * np.ones(n)
    ell = 5 * np.random.rand(n)
    # ell = 3 * np.ones(n)

    b = 20 * mu_max
    c = 40 * tf * mu_max
    d = np.ones(n)

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    # t_opt = maximize_shaping(b, c, d, ell, t0, tf, mu, alpha, w)
    # eval_shaping(t_opt, b, d, ell, tf, alpha, w)

    t_opt = maximize_shaping_int(b, c, d, ell, t0, tf, mu, alpha, w)
    eval_shaping_int(t_opt, b, d, ell, tf, alpha, w)
    # # eval_shaping_int(lambda s: [b * (s < t_opt[j]) for j in range(n)], b, d, ell, tf, alpha, w)

    # tt = np.arange(t0, tf, 1)
    # yy = np.zeros(len(tt))
    # for i in range(n):
    #     for k in range(len(tt)):
    #         yy[k] = f_int(tt[k], i, t0, tf, alpha, w, d, ell, b)
    #     plt.plot(tt, yy)
    # plt.show()


if __name__ == '__main__':
    main()
