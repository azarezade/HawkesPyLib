# __author__ = 'Ali_Zarezade'

import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

from numpy.linalg import inv
from scipy.linalg import logm, expm2
from scipy.optimize import brentq, newton_krylov, anderson, least_squares
import scipy.integrate as integrate
from event_generation import *
from activity_maximization import psi


def g(t, tf, alpha, w):
    return psi(tf - t, alpha, w)


def g_int(t, tf, alpha, w):
    n = alpha.shape[0]
    I = np.eye(n)
    alpha_w = alpha - w * I
    alpha_w_inv = inv(alpha_w)
    if t > tf:
        return np.zeros((n, n))
    elif t == tf:
        return I
    else:
        return I - alpha.dot(alpha_w_inv).dot(I - expm2(alpha_w * (tf - t)))


# def f(s, i, tf, alpha, w, ell):
#     """
#     f_i(s_i) = nu
#     sum s_i = n*tf - c/b
#     """
#     n = alpha.shape[0]
#     return 2 * (g(s[i], tf, alpha, w)[:, i].dot(ell - sum([g_int(s[k], tf, alpha, w)[i, k] for k in range(n)])))


def f(s, tf, alpha, w, ell, b):
    n = alpha.shape[0]
    val = np.zeros(n)
    for i in range(n):
        val[i] = 2 * g(s[i], tf, alpha, w)[:, i].dot(ell - b * sum([g_int(s[k], tf, alpha, w)[i, k] for k in range(n)]))
    return val


def f_int(s, i, t0, tf, alpha, w, d, ell, b):
    """
    f_i(s_i) = nu
    sum s_i = n*tf - c/b
    """
    return


def maximize_shaping(b, c, ell, t0, tf, alpha, w, tol=5*1e-1):
    """
    Solve the following optimization: TBD...
    """
    n = alpha.shape[0]
    t_opt = tf * np.ones(n)

    r = max(np.abs(np.linalg.eig(alpha)[0]))
    print("spectral radius = {}".format(r))
    if r > w:
        raise Exception("spectral radius r={} is greater that w={}".format(r, w))

    boundaries = np.concatenate([f(t0 * np.ones(n), tf, alpha, w, ell, b), f(tf * np.ones(n), tf, alpha, w, ell, b)], axis=0)
    lb = min(boundaries)
    ub = max(boundaries)
    t_0 = tf * 0.99 * np.ones(n)
    print("ub={} \t lb={}".format(ub, lb))

    # TODO: revise the following condition!
    res = least_squares(lambda s: b * sum([g_int(s[i], tf, alpha, w)[:, i] for i in range(n)]) - ell, t_0,
                        bounds=(0, 100), verbose=1,  loss='cauchy', xtol=1e-2)
    t_opt = res.x
    if n*tf - sum(t_opt) < c/b:
        print('budget inequality constraint is inactive')
        return t_opt
    else:
        prev_tol = 0
        # TODO: revise the following condition, as well!
        while abs(sum(t_opt) + c/b - n*tf) > tol:
            m = (ub + lb) / 2.0  # m = nu
            res = least_squares(lambda s: f(s, tf, alpha, w, ell, b) - m, t_0, bounds=(0, 100), verbose=1, xtol=1e-2)
            t_opt = res.x
            t_0 = t_opt
            print('ub={} \t lb={} \t tol={} \n t_opt={}'.format(ub, lb, n*tf - sum(t_opt) - c/b, t_opt))
            print("optimal = {}".format(eval_shaping(t_opt, b, ell, tf, alpha, w)))
            if abs(prev_tol - (n*tf - sum(t_opt) - c/b)) < 1e-3:
                return t_opt
            if n*tf - sum(t_opt) < c/b:
                ub = m
            else:
                lb = m
            prev_tol = n*tf - sum(t_opt) - c/b
        return t_opt


def eval_shaping(s, b, ell, tf, alpha, w):
    n = alpha.shape[0]
    I = np.eye(n)
    Aw = alpha - w * I

    int_total = np.zeros(n)
    for i in range(n):
        # int_u_g = (alpha.dot(inv(Aw)).dot(expm2(Aw * tf) - expm2(Aw * (tf - s[i]))))[:, i] + float(s[i] == tf) * I[:, i]
        int_u_g = (I - alpha.dot(inv(Aw)).dot(I - expm2(Aw * (tf - s[i]))))[:, i]
        int_total += b * int_u_g

    obj = np.linalg.norm(int_total - ell) / n
    # print(int_total)
    # print(ell)
    # print(obj)
    return obj


def eval_shaping_int(s, b, d, ell, tf, alpha, w):
    n = alpha.shape[0]
    I = np.eye(n)
    Aw = alpha - w * I
    Awi = inv(Aw)
    Awi2 = Awi.dot(Awi)

    integral = np.zeros(n)
    for i in range(n):
        u_psi = (I * s[i] + alpha.dot(Awi2).dot(expm2(Aw * (tf - s[i])) - expm2(Aw * tf)) - s[i] * alpha.dot(Awi))[:, i]
        integral[i] = b * u_psi.dot(d)

    obj = np.linalg.norm(integral - ell)
    print(integral)
    print(ell)
    print(obj)
    return obj


def main():
    np.random.seed(5)
    t0 = 0
    tf = 100
    n = 16
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 2

    b = 100 * mu_max
    c = 1 * n * tf * mu_max
    # d = np.ones(n)

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    ell = 110 * g_int(0, tf, alpha, w).dot(mu_max * np.ones(n))
    # ell = np.ones(n) + np.array([0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.06])
    # int_mu_g = g_int(0, tf, alpha, w).dot(mu)

    t_opt = maximize_shaping(b, c, ell, t0, tf, alpha, w)
    print("optimal = {}, uniform = {}".format(eval_shaping(t_opt, b, ell, tf, alpha, w),
                                              eval_shaping(np.zeros(n), c/(n*tf), ell, tf, alpha, w)))
    # print(t_opt)
    print(n*tf - sum(t_opt), c/b)

    # t_opt = maximize_shaping_int(b, c, d, ell, t0, tf, mu, alpha, w)
    # eval_shaping_int(t_opt, b, d, ell, tf, alpha, w)
    # # eval_shaping_int(lambda s: [b * (s < t_opt[j]) for j in range(n)], b, d, ell, tf, alpha, w)

    # t = np.arange(t0, tf, 1)
    # y = np.zeros((n, n, len(t)))
    # for k in range(len(t)):
    #     y_matrix = g_int(t[k], tf, alpha, w)
    #     # y_matrix = psi_int(t[k], t0, tf, alpha, w)
    #     for i in range(n):
    #         for j in range(n):
    #             y[i, j, k] = y_matrix[i, j]
    # for i in range(n):
    #     for j in range(n):
    #         plt.plot(t, y[i, j, :])
    # plt.show()

    # t = np.arange(t0, tf, 2)
    # y = np.zeros((n, len(t)))
    # for i in range(n):
    #     for k in range(len(t)):
    #         y[:, k] = f(np.ones(n)*t[k], tf, alpha, w, ell, b)
    #     plt.plot(t, y[i, :])
    # plt.show()


if __name__ == '__main__':
    main()
