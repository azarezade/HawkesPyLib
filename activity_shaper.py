# __author__ = 'alza'

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from scipy.linalg import expm
from scipy.optimize import fsolve
import scipy.integrate as integrate
from generator import generate_model, generate_events


def psi(t, alpha, w=1):
    """
    Consider Hawkes process lambda(t)=mu(t)+int_{-inf}^{t} g(t-s) alpha dN(s),
    the expected intensity is eta(t)=convolve(psi(t), mu(t)).     
    """
    n = alpha.shape[0]
    if t < 0:
        return np.zeros((n, n))
    elif t == 0:
        return np.eye(n) + alpha
    else:
        return alpha.dot(expm((alpha - w * np.eye(n)) * t))


def psi_sum(t, d, i, alpha, w=1):
    """
    Evaluate sum_j psi_ji(t)*d_j 
    """
    return psi(t, alpha, w)[:, i].dot(d)


def ipsi(t, t0, tf, alpha, w=1):
    """
    Integral of psi at time t, int_psi(t) = int_{t0}^{tf} psi(s-t) ds      
    """
    n = alpha.shape[0]
    if tf - t < 0:
        return np.zeros(n)
    else:
        return np.eye(n) + alpha.dot(inv(alpha - w * np.eye(n))).dot(expm((alpha - w * np.eye(n)) * (tf - t)) - np.eye(n))


def weighted_activity(t, u, d, t0, tf, alpha, w=1, tol=1e-1):
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
        tol (float): tolerance for attaining the b*sum(t)=c constraint
    """
    integral = 0
    n = alpha.shape[0]
    for i in range(n):
        integral += integrate.quad(lambda s: u(s, i) * psi_sum(t - s, d, i, alpha, w=1)
                                   , t0, tf)[0]
        integral += u(t, i)

    return integral


def max_weighted_activity(b, c, d, t0, tf, alpha, w=1, tol=1e-1):
    """
    Solve the following optimization:
        maximize    d^T eta(tf)
        subject to  0< mu_i(t) < b
                    int_t0^tf sum_i mu_i(s) ds = c
    """
    n = alpha.shape[0]
    ub = psi(0, alpha).sum(axis=0).max()
    lb = 0
    t = np.zeros(n)

    if (n * tf) * b < c:
        print("equality constraint can't be attained")
        return t

    print((n * tf - sum(t)) * b, c)
    while abs((n * tf - sum(t)) * b - c) > tol:  # or ub - lb > tol !?
        # print("upper bound={}, lower bound={}".format(ub, lb))
        m = (ub + lb) / 2
        for i in range(n):
            t[i] = fsolve(lambda s: psi_sum(tf-s, d, i, alpha) - m, tf*0.9)
            # print("the value of function at root is {}".format(psi_sum(tf - t[i], d, i, alpha) - m))
        # print((n * tf - sum(t)) * b, c)
        # print(t)
        if (n * tf - sum(t)) * b > c:
            lb = m
        else:
            ub = m
    # print("the value of function at root is {}".format(psi_sum(tf - t[i], d, i, alpha) - m))
    # print("sum(t)*b={}, c={}".format(sum(t) * b, c))
    return t


def main():
    # np.random.seed(200)
    t0 = 0
    tf = 100
    n = 5
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    # times, users = generate_events(mu, alpha, tf)

    # print(mu)
    # print(len(times))

    # t = np.arange(0, tf, 0.1)
    # y = np.zeros(t.shape)
    # for i in range(tf*10):
    #     y[i] = psi_sum(tf - t[i], d=np.ones(n), i=1, alpha=alpha)
    # plt.plot(t, y)
    # plt.show()

    b = 1 * mu_max
    c = 1 * tf * mu_max
    d = np.ones(n)
    t0 = 0
    tf = tf
    t_star = max_weighted_activity(b, c, d, t0, tf, alpha)
    print("t_star = ", t_star)

    def u_star(t, i):
        return b * (t > t_star[i])

    def u_star_vec(t):
        temp = np.zeros(n)
        for i in range(n):
            temp[i] = b * (t > t_star[i])
        return temp

    print("b=", b, "c=", c)
    print("base intensity budget {} results {}  in eta ".
          format(sum(mu) * tf, inv(np.eye(n) - alpha).dot(mu).sum()))
    print("optimal intensity budget {} results {}  in eta ".
          format(c, weighted_activity(tf, u_star, d, t0, tf, alpha)))

    def u_0(t, i):
        return mu_max / n
    print(weighted_activity(tf, u_0, d, t0, tf, alpha))

    def u_0_vec(t):
        return (mu_max / n) * np.ones(n)

    times_optimal, users_optimal = generate_events(t0, 10000, mu, alpha, u_star_vec)
    times_poisson, users_poisson = generate_events(t0, 10000, mu, alpha, u_0_vec)

    print(c, len(times_optimal))
    print(mu_max*tf, len(times_poisson))


if __name__ == '__main__':
    main()
