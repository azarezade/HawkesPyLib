# __author__ = 'Ali_Zarezade'

# # Set limit to the number of cores:
# import mkl
# mkl.set_num_threads(n)

import matplotlib as mpl; mpl.use('Agg')
from numpy.linalg import inv
from scipy.linalg import logm, expm
from scipy.optimize import fsolve, brentq
import scipy.integrate as integrate
from event_generation import *


def psi(t, alpha, w=1.0):
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


def psi_int(t, t0, tf, alpha, w=1):
    """
    Integral of psi at time t, int_psi(t) = int_{t0}^{tf} psi(s-t) ds
    """
    n = alpha.shape[0]
    I = np.eye(n)
    if (t > tf) or (t < t0):
        return np.zeros((n, n))
    return I + alpha.dot(inv(alpha - w * I)).dot(expm((alpha - w * I) * (tf - t)) - I)


def activity_max(b, c, d, t0, tf, alpha, w=1.0, tol=1e-1):
    """
    Solve the following optimization:
        maximize    d^T eta(tf)
        subject to  0< mu_i(t) < b
                    int_t0^tf sum_i mu_i(s) ds = c
    where
        tol (float): tolerance for attaining the b*sum(t)=c constraint
    """
    n = alpha.shape[0]

    r = max(np.abs(np.linalg.eig(alpha)[0]))
    print("spectral radius = {}".format(r))
    if r > w:
        raise Exception("spectral radius r={} is greater that w={}".format(r, w))

    if (n * tf) * b < c:
        print("equality constraint can't be attained")
        return np.ones(n) * tf

    t = np.zeros(n)
    # lb = 0
    lb = min(np.dot(d, psi(tf - t0, alpha, w)))
    ub = max(np.dot(d, psi(tf - tf, alpha, w)))
    while abs((n * tf - sum(t)) * b - c) > tol:
        m = (ub + lb) / 2
        for i in range(n):
            if psi(0, alpha, w)[:, i].dot(d) < m:
                t[i] = tf
            else:
                (t[i], rep) = brentq(lambda s: psi(tf - s, alpha, w)[:, i].dot(d) - m, t0, tf, full_output=True)
                k = i
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(i) for i in t], psi(tf-t[k], alpha, w)[:, k].dot(d) - m, (n * tf - sum(t)) * b - c))
        if (n * tf - sum(t)) * b > c:
            lb = m
        else:
            ub = m
    return t


def eval_activity_max(t, u, d, t0, tf, alpha, w=1.0):
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


def activity_max_int(b, c, d, t0, tf, alpha, w=1, tol=1e-1):
    """
    Solve the following optimization:
        maximize    int_t0^tf d^T eta(tf)
        subject to  0< mu_i(t) < b
                    int_t0^tf sum_i mu_i(s) ds = c
    where
        tol (float): tolerance for attaining the b*sum(t)=c constraint
    """
    n = alpha.shape[0]

    r = max(np.abs(np.linalg.eig(alpha)[0]))
    print("spectral radius = {}".format(r))
    if r > w:
        raise Exception("spectral radius r={} is greater that w={}".format(r, w))

    if (n * tf) * b < c:
        print("equality constraint can't be attained")
        return np.zeros(n)

    if c < b * tf:
        print("the algorithm may not converge, c={} < b*tf={}".format(c, b * tf))

    t = np.zeros(n)
    # lb = 0
    lb = min(np.dot(d, psi_int(tf, t0, tf, alpha, w)))
    ub = max(np.dot(d, psi_int(t0, t0, tf, alpha, w)))
    while abs(sum(t) * b - c) > tol:
        m = (ub + lb) / 2
        for i in range(n):
            if psi_int(0, t0, tf, alpha, w)[:, i].dot(d) < m:
                t[i] = 0
            else:
                (t[i], re) = brentq(lambda s: psi_int(s, t0, tf, alpha, w)[:, i].dot(d) - m, 0, tf, full_output=True)
                k = i
        print("ub={:.4f} \t lb={:.4f} \t t_star={} precision={} diff={}".
              format(ub, lb, [int(i) for i in t], psi_int(t[k], t0, tf, alpha, w)[:, k].dot(d) - m, sum(t) * b - c))
        if sum(t) * b > c:
            lb = m
        else:
            ub = m
        # TODO: the following stopping cond is add-hoc, change it!
        if ub - lb < 1e-5:
            if sum(t) * b - c < 0.1 * c:
                return t
            else:
                raise Exception("not converged ub={}, lb={}".format(ub, lb))
    return t


def eval_activity_max_int(u, d, t0, tf, alpha, w=1):
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


def main():
    # np.random.seed(0)
    t0 = 0
    tf = 100
    n = 64
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 5

    b = 10 * mu_max
    c = 10 * tf * mu_max
    d = np.ones(n)

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    # sio.savemat('./dataset/pydata-64.mat', {'T': tf, 'N': n, 'w': w, 'mu': mu, 'alpha': alpha, 'c': d, 'C': c / tf})

    # activity_max(b, c, d, t0, tf, alpha, w)

    # activity_max_int(b, c, d, t0, tf, alpha, w)

    # r = max(np.abs(np.linalg.eig(alpha)[0]))

    # t = np.arange(t0, tf, 1)
    # y = np.zeros((n, len(t)))
    # for k in range(len(t)):
    #     y_matrix = psi(t[k], alpha, w)
    #     # y_matrix = psi_int(t[k], t0, tf, alpha, w)
    #     for i in range(n):
    #         y[i, k] = y_matrix[:, i].dot(d)
    # for i in range(n):
    #     plt.plot(t, y[i, :])
    # plt.show()


if __name__ == '__main__':
    main()
