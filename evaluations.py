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


def compare_weighted_activity(mat_address):
    t0, tf, n, sparsity, mu_max, alpha_max, mu, alpha, b, c, d, u_cam = load_mat(mat_address)
    t_star = maximize_weighted_activity(b, c, d, t0, tf, alpha)

    def u_deg(t):
        deg = np.zeros(n)
        for i in range(n):
            deg[i] = np.count_nonzero(alpha[i, :])
        return (deg / sum(deg)) * (c / tf)

    def u_optimal(t):
        return [b * (t > t_star[i]) for i in range(n)]

    def u_mehrdad(t):
        return [u_cam[i] for i in range(n)]

    def u_prk(t):
        G = nx.from_numpy_matrix(alpha)
        pr = nx.pagerank(G)
        weight = np.asanyarray(list(pr.values()))
        return (weight / sum(weight)) * (c / tf)

    def u_uniform(t):
        return [c / (tf * n) for i in range(n)]

    obj_uniform = eval_weighted_activity(tf, u_uniform, d, t0, tf, alpha, w)
    obj_deg = eval_weighted_activity(tf, u_deg, d, t0, tf, alpha, w)
    obj_prk = eval_weighted_activity(tf, u_prk, d, t0, tf, alpha, w)
    obj_mehrdad = eval_weighted_activity(tf, u_mehrdad, d, t0, tf, alpha, w)
    obj_optimal = eval_weighted_activity(tf, u_optimal, d, t0, tf, alpha, w)
    print("obj_uniform={} \t\t ".format(obj_uniform))
    print("obj_deg={} \t\t ".format(obj_deg))
    print("obj_prk={} \t\t ".format(obj_prk))
    print("obj_mehrdad={} \t\t ".format(obj_mehrdad))
    print("obj_optimal={} \t\t ".format(obj_optimal))


    times_base, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha)
    times_mehrdad, _ = generate_events(t0=t0, tf=tf, mu=mu + u_cam, alpha=alpha)
    times_optimal, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_optimal)

    def count_events(times, a, b):
        k = 0
        for i in range(len(times)):
            if a < times[i] < b:
                k += 1
        return k

    print("base \t total_event={} \t last_events={}".
          format(len(times_base), count_events(times_base, 0.9 * tf, tf)))
    print("mehrdad \t total_event={} \t last_events={}".
          format(len(times_mehrdad), count_events(times_mehrdad, 0.9 * tf, tf)))
    print("optimal \t total_event={} \t last_events={}".
          format(len(times_optimal), count_events(times_optimal, 0.9 * tf, tf)))

    return


def compare_int_weighted_activity(t0, tf, b, c, d, w, n, sparsity, mu_max, alpha_max):
    """ Compare the proposed method with poisson baseline. """
    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    t_star = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)

    def u_optimal(t):
        return [b * (t < t_star[i]) for i in range(n)]

    def u_uniform(t):
        return [c / (tf * n) for i in range(n)]

    def u_prk(t):
        G = nx.from_numpy_matrix(alpha)
        pr = nx.pagerank(G)
        weight = np.asanyarray(list(pr.values()))
        return (weight / sum(weight)) * (c / tf)

    def u_deg(t):
        deg = np.zeros(n)
        for i in range(n):
            deg[i] = np.count_nonzero(alpha[i, :])
        return (deg / sum(deg)) * (c / tf)

    # obj_deg = eval_int_weighted_activity(u_deg, d, t0, tf, alpha, w)
    # obj_prk = eval_int_weighted_activity(u_prk, d, t0, tf, alpha, w)
    obj_uniform = eval_int_weighted_activity(u_uniform, d, t0, tf, alpha, w)
    obj_optimal = eval_int_weighted_activity(u_optimal, d, t0, tf, alpha, w)
    # print("obj_deg={} \t\t ".format(obj_deg))
    # print("obj_prk={} \t\t ".format(obj_prk))
    print("obj_uniform={} \t\t ".format(obj_uniform))
    print("obj_optimal={} \t\t ".format(obj_optimal))

    times_base, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha)
    times_poisson, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_uniform)
    times_optimal, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_optimal)

    print("poisson \t num of event={} \t increase(%)={}".
          format(len(times_poisson), 100 * (len(times_poisson) - len(times_base)) / len(times_base)))
    print("optimal \t num of event={} \t increase(%)={}".
          format(len(times_optimal), 100 * (len(times_optimal) - len(times_base)) / len(times_base)))
    return

def compare_batch_int_weighted_activity(num):
    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    t_star = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)

    def u_optimal(t):
        return [b * (t < t_star[i]) for i in range(n)]

    def u_uniform(t):
        return [c / (tf * n) for i in range(n)]

    def u_prk(t):
        G = nx.from_numpy_matrix(alpha)
        pr = nx.pagerank(G)
        weight = np.asanyarray(list(pr.values()))
        return (weight / sum(weight)) * (c / tf)

    def u_deg(t):
        deg = np.zeros(n)
        for i in range(n):
            deg[i] = np.count_nonzero(alpha[i, :])
        return (deg / sum(deg)) * (c / tf)

    # obj_deg = eval_int_weighted_activity(u_deg, d, t0, tf, alpha, w)
    # obj_prk = eval_int_weighted_activity(u_prk, d, t0, tf, alpha, w)
    # obj_uniform = eval_int_weighted_activity(u_uniform, d, t0, tf, alpha, w)
    # obj_optimal = eval_int_weighted_activity(u_optimal, d, t0, tf, alpha, w)
    # print("obj_deg={} \t\t ".format(obj_deg))
    # print("obj_prk={} \t\t ".format(obj_prk))
    # print("obj_uniform={} \t\t ".format(obj_uniform))
    # print("obj_optimal={} \t\t ".format(obj_optimal))

    event_count_deg = 0
    event_count_uniform = 0
    event_count_optimal = 0
    for ii in range(num):
        # times_base, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha)
        times_deg, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_deg)
        times_uniform, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_uniform)
        times_optimal, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_optimal)
        event_count_deg += len(times_deg)
        event_count_uniform += len(times_uniform)
        event_count_optimal += len(times_optimal)
    print("deg={} \t uniform={} \t optimal={}".format(event_count_deg, event_count_uniform, event_count_optimal))
    return


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