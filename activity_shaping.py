# __author__ = 'Ali_Zarezade'

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from numpy.linalg import inv
from scipy.linalg import logm, expm
from scipy.optimize import fsolve
import scipy.integrate as integrate
from event_generation import *

# TODO: correct this line 'ub = psi(0, alpha).sum(axis=0).max()'


def psi(t, alpha, w=1):
    """
    Consider Hawkes process lambda(t)=mu(t)+int_{-inf}^{t} g(t-s) alpha dN(s),
    the expected intensity is eta(t)=convolve(psi(t), mu(t)).     
    """
    n = alpha.shape[0]
    I = np.eye(n)
    if t < 0:
        return np.zeros((n, n))
    elif t == 0:
        return I + alpha
    else:
        return alpha.dot(expm((alpha - w * I) * t))


def eval_weighted_activity(t, u, d, t0, tf, alpha, w=1):
    """
    Evaluate the following objective function
      d^T eta(tf) = int_0^tf sum_i mu_i(s)*(sum_j psi_ji(tf-s)*d_j) ds

    Arguments:
        t (float): the time at which the weighted activity wil be evaluated
        u (func): control base intensity function where u(t, i) is control of i'th user at time t
        d (ndarray): weigh of users expected intensity
        t0 (float): initial time
        tf (float): terminal time
        alpha (ndarray): influence matrix
        w (float): weight of exponential kernel
    """
    integral = 0
    n = alpha.shape[0]
    for i in range(n):
        integral += u(t)[i] + integrate.quad(lambda s: u(s)[i] * psi(t - s, alpha, w)[:, i].dot(d), t0, tf)[0]
    return integral


def maximize_weighted_activity(b, c, d, t0, tf, alpha, w=1, tol=1e-1):
    """
    Solve the following optimization:
        maximize    d^T eta(tf)
        subject to  0< mu_i(t) < b
                    int_t0^tf sum_i mu_i(s) ds = c
    where
        tol (float): tolerance for attaining the b*sum(t)=c constraint
    """
    n = alpha.shape[0]
    ub = d.dot(psi(0, alpha)).max()
    lb = 0
    t = np.zeros(n)

    if (n * tf) * b < c:
        print("equality constraint can't be attained")
        return t

    while abs((n * tf - sum(t)) * b - c) > tol:
        m = (ub + lb) / 2
        for i in range(n):
            t[i] = fsolve(lambda s: psi(tf - s, alpha, w)[:, i].dot(d) - m, tf * 0.99)

        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(i) for i in t], psi(t[i], alpha, w)[:, i].dot(d) - m, (n * tf - sum(t)) * b - c))
        # print("ub={:.4f} \t lb={:.4f} \t t_star={}".format(ub, lb, [int(i) for i in t]))
        if (n * tf - sum(t)) * b > c:
            lb = m
        else:
            ub = m
    return t


def psi_int(t, t0, tf, alpha, w=1):
    """
    Integral of psi at time t, int_psi(t) = int_{t0}^{tf} psi(s-t) ds
    """
    n = alpha.shape[0]
    I = np.eye(n)
    if (t > tf) or (t < t0):
        return np.zeros((n, n))
    return I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * (tf - t)) - I)


def eval_int_weighted_activity(u, d, t0, tf, alpha, w=1):
    """
    Evaluate the following objective function
      int_t0^tf d^T eta(s) ds = int_t0^tf sum_i mu_i(s)*(sum_j int_t0^tf psi_ji(t-s)*d_j dt) ds
    Arguments:
        u (func): control base intensity function where u(t, i) is control of i'th user at time t
        d (ndarray): weigh of users expected intensity
        t0 (float): initial time
        tf (float): terminal time
        alpha (ndarray): influence matrix
        w (float): weight of exponential kernel
    """
    integral = 0
    n = alpha.shape[0]
    for i in range(n):
        integral += integrate.quad(lambda s: u(s)[i] * psi_int(s, t0, tf, alpha, w)[:, i].dot(d), t0, tf)[0]
    return integral


def maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w=1, tol=1e-1):
    """
    Solve the following optimization:
        maximize    int_t0^tf d^T eta(tf)
        subject to  0< mu_i(t) < b
                    int_t0^tf sum_i mu_i(s) ds = c
    where
        tol (float): tolerance for attaining the b*sum(t)=c constraint
    """
    n = alpha.shape[0]
    ub = d.dot(psi_int(0, t0, tf, alpha, w)).max()
    lb = 0
    t = np.zeros(n)

    if (n * tf) * b < c:
        print("equality constraint can't be attained")
        return t

    while abs(sum(t) * b - c) > tol:
        m = (ub + lb) / 2
        for i in range(n):
            if psi_int(0, t0, tf, alpha, w)[:, i].dot(d) < m:
                t[i] = 0
            else:
                t[i] = fsolve(lambda s: psi_int(s, t0, tf, alpha, w)[:, i].dot(d) - m, tf * 0.9)
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(i) for i in t], psi_int(t[i], t0, tf, alpha, w)[:, i].dot(d) - m, sum(t) * b - c))
        if sum(t) * b > c:
            lb = m
        else:
            ub = m
    return t


def main():
    # np.random.seed(100)
    t0 = 0
    tf = 100
    n = 50
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 0.6

    b = 1 * mu_max
    c = 1 * tf * mu_max
    d = np.ones(n)

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    t_star = maximize_weighted_activity(b, c, d, t0, tf, alpha)

    def u_star(t):
        return [b * (t > t_star[i]) for i in range(n)]

    print("base intensity budget={} results {}  in eta ".
          format(sum(mu) * tf, inv(np.eye(n) - alpha).dot(mu).sum()))
    print("optimal intensity budget={} results {}  in eta ".
          format(c, eval_weighted_activity(tf, u_star, d, t0, tf, alpha)))

    # tt = np.arange(t0, tf, 1)
    # yy = np.zeros(len(tt))
    # # for i in range(n):
    # #     for j in range(n):
    # #         for k in range(len(tt)):
    # #             tmp = psi_int(tt[k], t0, tf, alpha, w=1)
    # #             # tmp = psi(tf-tt[k], alpha, w=1)
    # #             yy[k] = tmp[i, j]
    # #         plt.plot(tt, yy)
    # for i in range(n):
    #     for k in range(len(tt)):
    #         yy[k] = psi(tf-tt[k], alpha, w)[:, i].dot(d)
    #         # yy[k] = psi_int(tt[k], t0, tf, alpha, w)[:, i].dot(d)
    #     plt.plot(tt, yy)
    # plt.show()

    # t_star_int = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)
    #
    # def u_star_int(t):
    #     return [b * (t < t_star_int[i]) for i in range(n)]
    #
    # def u_poisson(t):
    #     return [c / (tf * n) for i in range(n)]
    #
    # print('optimal \t', eval_int_weighted_activity(u_star_int, d, t0, tf, alpha, w))
    # print('poisson \t', eval_int_weighted_activity(u_poisson, d, t0, tf, alpha, w))

if __name__ == '__main__':
    main()
