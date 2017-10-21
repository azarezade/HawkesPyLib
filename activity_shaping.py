# __author__ = 'Ali_Zarezade'

import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

# # Set limit to the number of cores:
# import mkl
# mkl.set_num_threads(n)

from numpy.linalg import inv
from scipy.linalg import logm, expm
from scipy.optimize import least_squares
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
        return I - alpha.dot(alpha_w_inv).dot(I - expm(alpha_w * (tf - t)))


def f(s, tf, alpha, w, ell, b):
    n = alpha.shape[0]
    val = np.zeros(n)
    for i in range(n):
        val[i] = 2 * g(s[i], tf, alpha, w)[:, i].dot(ell - b * sum([g_int(s[k], tf, alpha, w)[i, k] for k in range(n)]))
    return val


def f_new(s, tf, alpha, w, ell, b):
    n = alpha.shape[0]
    val = np.zeros(n)
    for i in range(n):
        val[i] = 2 * g(s[i], tf, alpha, w)[:, i].dot(ell - s[i+n] * sum([g_int(s[k], tf, alpha, w)[i, k] for k in range(n)]))
    return val


def f_int(s, i, t0, tf, alpha, w, d, ell, b):
    """
    f_i(s_i) = nu
    sum s_i = n*tf - c/b
    """
    return


def maximize_shaping(b, c, ell, t0, tf, alpha, w, tol=1e-4):
    """
    Solve the following optimization: TBD...
    """
    n = alpha.shape[0]
    r = max(np.abs(np.linalg.eig(alpha)[0]))
    print("spectral radius = {}".format(r))
    if r > w:
        raise Exception("spectral radius r={} is greater that w={}".format(r, w))

    x_0 = np.append(tf * 0.9 * np.ones(n), b * np.ones(n))
    res = least_squares(lambda s: sum([s[n+i] * g_int(s[i], tf, alpha, w)[:, i] for i in range(n)]) - ell, x_0,
                        bounds=(np.zeros(n+n), np.append(100*np.ones(n), b*np.ones(n))),
                        loss='cauchy', xtol=tol, verbose=1)
    x_opt = res.x
    t_opt = x_opt[:n]
    u_opt = x_opt[n:2*n]

    # opt_obj, opt_int = eval_shaping(t_opt, u_opt, ell, tf, alpha, w)
    # print(opt_obj, np.dot(tf - t_opt, u_opt), c)
    # print(t_opt, u_opt)

    if np.dot(tf - t_opt, u_opt) < c:
        print('Budget inequality constraint is inactive.\n')
    else:
        t_0 = tf * 0.99 * np.ones(n)
        res = least_squares(lambda s: sum([b * g_int(s[i], tf, alpha, w)[:, i] for i in range(n)]) - ell +
                            1e8 * (b*sum(tf-t_opt) > c), t_0, bounds=(0, 100), xtol=tol, verbose=1)
        t_opt = res.x
        u_opt = b*np.ones(n)
    return t_opt, u_opt


def eval_shaping(s, u, ell, tf, alpha, w):
    """
    Evaluate the least square objective for control intensity with i'th element:
    u[i] * ( t > s[i])
    """
    n = alpha.shape[0]
    I = np.eye(n)
    Aw = alpha - w * I
    int_total = np.zeros(n)
    for i in range(n):
        int_u_g = (I - alpha.dot(inv(Aw)).dot(I - expm(Aw * (tf - s[i]))))[:, i] * (s[i] < tf)
        int_total += u[i] * int_u_g
    obj = np.linalg.norm(int_total - ell) ** 2
    return obj, int_total


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
    print(integral)
    print(ell)
    print(obj)
    return obj


def main():
    np.random.seed(1)
    t0 = 0
    tf = 100
    n = 32
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 1
    b = 100 * mu_max
    c = 1 * n * tf * mu_max
    ell = 1 * np.array([0.0125, 0.0250, 0.0500, 0.7500] * int(n/4))

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    # alpha = 0.5 * (np.transpose(alpha) + alpha)
    sio.savemat('./data/pydata-lsq-32.mat', {'T': tf, 'N': n, 'w': w, 'mu': mu, 'alpha': alpha, 'C': c / tf, 'v': ell})

    t_opt, u_opt = maximize_shaping(b, c, ell, t0, tf, alpha, w)
    opt_obj, opt_int = eval_shaping(t_opt, u_opt, ell, tf, alpha, w)
    unf_obj, unf_int = eval_shaping(np.zeros(n), c / (n * tf) * np.ones(n), ell, tf, alpha, w)
    print("opt_obj = {}, unf_obj = {} \n opt_int = {} \n  unf_int = {},".format(opt_obj, unf_obj, opt_int, unf_int))
    print("opt_t = {}, \n opt_u = {}".format(t_opt, u_opt))
    print("opt_budget = {}, c = {}".format(np.dot(tf - t_opt, u_opt), c))

    # t = np.arange(0, tf, 2)
    # y = np.zeros((n, len(t)))
    # for i in range(n):
    #     for k in range(len(t)):
    #         y[:, k] = f(np.ones(n)*t[k], tf, alpha, w, ell, b)
    #     plt.plot(t, y[i, :])
    # plt.show()


if __name__ == '__main__':
    main()
