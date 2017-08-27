# __author__ = 'Ali_Zarezade'

import networkx as nx
# import numpy as np

from event_generation import *
from activity_shaping import *


def load_mat(path):
    mat_contents = sio.loadmat(path)
    t0 = float(mat_contents['t0'][0][0])
    tf = float(mat_contents['tf'][0][0])
    n = int(mat_contents['n'][0][0])
    w = int(mat_contents['w'][0][0])
    # sparsity = float(mat_contents['sparsity'][0][0])
    # mu_max = float(mat_contents['mu_max'][0][0])
    # alpha_max = float(mat_contents['alpha_max'][0][0])
    mu = mat_contents['mu'][:, 0]
    alpha = mat_contents['alpha']
    # b = float(mat_contents['b'][0][0])
    c = float(mat_contents['c'][0][0])
    d = mat_contents['d'][:, 0]
    lambda_cam = mat_contents['lambda_cam'][:, 0]
    return t0, tf, n, w, mu, alpha, c, d, lambda_cam



if __name__ == '__main__':
    # np.random.seed(100)
    t0 = 0
    tf = 100
    n = 50
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 0.6

    b = 10 * mu_max
    c = 1 * tf * mu_max
    d = np.ones(n)

    x = np.arange(0, 10, 0.1)
    f = x**2
    plt.plot(x, f)
    # plt.show()
    plt.savefig('/Users/alizarezade/Desktop/t1.pdf')

    g = x**0.5
    plt.clf()
    plt.plot(x, g)
    plt.savefig('/Users/alizarezade/Desktop/t2.pdf')
    # plt.show()

    # compare_weighted_activity('./data/mehrdad_shaping.mat')
    #
    # compare_int_weighted_activity(t0, tf, b, c, d, w, n, sparsity, mu_max, alpha_max)
    # compare_batch_int_weighted_activity(5)