# __author__ = 'Ali_Zarezade'

# # Set limit to the number of cores:
# import mkl
# mkl.set_num_threads(n)

import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

from numpy.linalg import inv
from scipy.linalg import logm, expm
from scipy.optimize import least_squares, minimize
import scipy.integrate as integrate
from event_generation import *
from activity_maximization import psi
from numpy.linalg import norm


def g_max(t, tf, alpha, w):
    return psi(tf - t, alpha, w)


def g_max_int(t, tf, alpha, w):
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


def g_ls(t, tf, alpha, w):
    """
    This is equal to g_max_int indeed!!
    """
    n = alpha.shape[0]
    I = np.eye(n)
    alpha_w = alpha - w * I
    alpha_w_inv = inv(alpha_w)
    if t > tf:
        return np.zeros((n, n))
    elif t == tf:
        return I
    else:
        return I + alpha.dot(alpha_w_inv).dot(expm(alpha_w * (tf - t)) - I)


def g_ls_int(t, tf, alpha, w):
    n = alpha.shape[0]
    I = np.eye(n)
    alpha_w = alpha - w * I
    alpha_w_inv = inv(alpha_w)
    alpha_w_inv2 = alpha_w_inv.dot(alpha_w_inv)
    return (I - alpha.dot(alpha_w_inv)) * t + alpha.dot(alpha_w_inv2).dot(expm(alpha_w * tf) - expm(alpha_w * (tf - t)))


def maximize_shaping(b, c, ell, t0, tf, alpha, w, tol=1e-4):
    """
    Solve the following optimization: TBD...
    """
    n = alpha.shape[0]
    r = max(np.abs(np.linalg.eig(alpha)[0]))
    # print("spectral radius = {}".format(r))
    if r > w:
        raise Exception("spectral radius r={} is greater that w={}".format(r, w))

    x_0 = np.append(tf * 0.99 * np.ones(n), b * np.ones(n))
    res = least_squares(lambda s: 1e5 * (np.dot(tf - s[:n], s[n:2*n]) > c) +
                        (sum([s[n+i] * g_max_int(s[i], tf, alpha, w)[:, i] for i in range(n)]) - ell),
                        x_0, bounds=(np.zeros(n+n), np.append(tf*np.ones(n), b*np.ones(n))), loss='linear', xtol=tol)
    x_opt = res.x
    t_opt = x_opt[:n]
    u_opt = x_opt[n:2*n]
    return t_opt, u_opt


def maximize_int_shaping(b, c, ell, t0, tf, alpha, w, tol=1e-5):
    """
    Solve the following optimization: TBD...
    """
    n = alpha.shape[0]
    r = max(np.abs(np.linalg.eig(alpha)[0]))
    # print("spectral radius = {}".format(r))
    if r > w:
        raise Exception("spectral radius r={} is greater that w={}".format(r, w))

    x_0 = np.append(tf * 0.05 * np.ones(n), b * np.ones(n))
    g_ls_0 = g_ls(0, tf, alpha, w)

    # res = least_squares(lambda s: 1e5 * (np.dot(s[:n], s[n:2*n]) > c) +
    #                     (sum([s[n+i] * s[i] * g_ls_0[:, i] for i in range(n)]) - ell), x_0,
    #                     bounds=(np.zeros(n+n), np.append(tf*np.ones(n), b*np.ones(n))), loss='linear')

    # res = least_squares(lambda s: 1e5 * (np.dot(s[:n], s[n:2 * n]) > c) +
    #                               (sum([s[n + i] * g_ls_int(s[i], tf, alpha, w)[:, i] for i in range(n)]) - ell), x_0,
    #                     bounds=(np.zeros(n + n), np.append(tf * np.ones(n), b * np.ones(n))), loss='linear')

    # bnds = [(0,tf*(i<n)+b*(i>=n)) for i in range(2*n)]
    # cons = ({'type': 'ineq', 'fun': lambda s: - np.dot(s[:n], s[n:2*n]) + c})
    # res = minimize(lambda s: norm((sum([s[n+i] * s[i] * g_ls_0[:, i] for i in range(n)]) - ell))**2,
    #                x_0, method = 'SLSQP', bounds=bnds, constraints=cons, options = {'disp': True})

    # x_opt = res.x
    # t_opt = x_opt[:n]
    # u_opt = x_opt[n:2*n]
    # return t_opt, u_opt

    x_0 = tf * 0.1 * np.ones(n)
    res = least_squares(lambda s: 1e5 * (np.sum(s) > c) +
                                  (sum([s[i] * g_ls_0[:, i] for i in range(n)]) - ell), x_0,
                        bounds=(np.zeros(n), b * tf * np.ones(n)), loss='linear')
    x_opt = res.x
    t_opt = 10 * np.ones(n)
    u_opt = x_opt / 10
    return t_opt, u_opt



def eval_shaping(s, u, ell, tf, alpha, w):
    # TODO: edit eval_shaping and use g_max function
    """
        Evaluate the least square objective for control intensity with i'th element:
        u[i] * ( t > s[i])
    """
    n = alpha.shape[0]
    I = np.eye(n)
    Aw = alpha - w * I
    integral = np.zeros(n)
    for i in range(n):
        int_g = (I - alpha.dot(inv(Aw)).dot(I - expm(Aw * (tf - s[i]))))[:, i] * (s[i] < tf)
        integral += u[i] * int_g
    obj = np.linalg.norm(integral - ell) ** 2
    return obj, integral


def eval_int_shaping(s, u, ell, tf, alpha, w):
    """
    Evaluate the least square objective for control intensity with i'th element:
    u[i] * ( t < s[i])
    """
    n = alpha.shape[0]
    I = np.eye(n)
    Aw = alpha - w * I
    Awi = inv(Aw)
    Awi2 = Awi.dot(Awi)

    integral = np.zeros(n)
    for i in range(n):
        int_g = ((I - alpha.dot(Awi)) * s[i] - alpha.dot(Awi2).dot(expm(Aw * (tf - s[i])) - expm(Aw * tf)))[:, i]
        integral += u[i] * int_g
    obj = np.linalg.norm(integral - ell) ** 2
    return obj, integral


def main():
    np.random.seed(9)
    t0 = 0
    tf = 100
    n = 64
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 2
    b = 100 * mu_max
    c = 8 * n * tf * mu_max

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    # alpha = 0.5 * (np.transpose(alpha) + alpha)

    # ell = 2 * np.array([0.2, 0.2, 0.6, 0.8] * int(n / 4))
    # ell = ell - np.dot(g_max_int(0, tf, alpha, w), mu)
    # t_opt, u_opt = maximize_shaping(b, c, ell, t0, tf, alpha, w)
    # obj_opt, int_opt = eval_shaping(t_opt, u_opt, ell, tf, alpha, w)
    # obj_unf, int_unf = eval_shaping(np.zeros(n), c / (n * tf) * np.ones(n), ell, tf, alpha, w)
    # print("obj_opt = {}, obj_unf = {}".format(obj_opt, obj_unf))
    # print("ell = {}".format(ell))
    # print("int_opt = {} \n  int_unf = {},".format(int_opt, int_unf))
    # print("opt_t = {}, \n opt_u = {}".format(t_opt, u_opt))
    # print("opt_budget = {}, c = {}".format(np.dot(tf - t_opt, u_opt), c))

    ell = 10 * np.array([0.2, 0.4, 0.6, 0.8] * int(n / 4))
    # # ell = ell - np.dot(g_ls_int(tf, tf, alpha, w), mu)
    # # print(ell)
    t_opt, u_opt = maximize_int_shaping(b, c, ell, t0, tf, alpha, w)
    obj_opt, int_opt = eval_int_shaping(t_opt, u_opt, ell, tf, alpha, w)
    obj_unf, int_unf = eval_int_shaping(tf*np.ones(n), c / (n * tf) * np.ones(n), ell, tf, alpha, w)
    print("obj_opt = {}, obj_unf = {}".format(obj_opt, obj_unf))
    print("ell = {}".format(ell))
    print("int_opt = {} \n  int_unf = {},".format(int_opt, int_unf))
    print("opt_t = {}, \n opt_u = {}".format(t_opt, u_opt))
    print("opt_budget = {}, c = {}".format(np.dot(t_opt, u_opt), c))

    # x_opt = maximize_int_shaping(b, c, ell, t0, tf, alpha, w)
    # obj_opt, int_opt = eval_int_shaping(x_opt/0.1, 0.1*np.ones(n), ell, tf, alpha, w)
    # obj_unf, int_unf = eval_int_shaping(tf * np.ones(n), c / (n * tf) * np.ones(n), ell, tf, alpha, w)
    # print("obj_opt = {}, obj_unf = {}".format(obj_opt, obj_unf))
    # print("ell = {}".format(ell))
    # print("int_opt = {} \n  int_unf = {},".format(int_opt, int_unf))
    # print("opt_x = {}".format(x_opt))
    # print("opt_budget = {}, c = {}".format(sum(x_opt), c))

    # t = np.arange(0, tf, 2)
    # y = np.zeros((n, len(t)))
    # for i in range(n):
    #     for k in range(len(t)):
    #         y[:, k] = f(np.ones(n)*t[k], tf, alpha, w, ell, b)
    #     plt.plot(t, y[i, :])
    # plt.show()

    # t = np.arange(0, tf, 2)
    # y = np.zeros((n, n, len(t)))
    # for k in range(len(t)):
    #     y[:,:,k] = g_ls(t[k], tf, alpha, w)
    # for i in range(n):
    #     for j in range(n):
    #         plt.plot(t, y[i,j,:])
    # plt.show()

    # sio.savemat('./data/pydata-lsq-'+str(n)+'.mat',
    #             {'T': tf, 'N': n, 'w': w, 'mu': mu, 'alpha': alpha, 'C': c / tf, 'v': ell})


if __name__ == '__main__':
    main()
