# __author__ = 'Ali_Zarezade'

# # Set limit to the number of cores:
# import mkl
# mkl.set_num_threads(n)

import pickle
import numpy as np
import networkx as nx
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

from activity_maximization import maximize_weighted_activity, maximize_int_weighted_activity, eval_weighted_activity, eval_int_weighted_activity
from event_generation import generate_model, generate_events
from activity_shaping import eval_shaping, maximize_shaping, g_int


def u_deg(t, tf, c, deg):
    return (deg / sum(deg)) * (c / tf)


def u_prk(t, tf, c, weight):
    return (weight / sum(weight)) * (c / tf)


def u_unf(t, tf, c, n):
    return [c / (tf * n) for k in range(n)]


def u_opt_dec(t, t_opt, n, b):
    return [b * (t < t_opt[j]) for j in range(n)]


def u_opt_inc(t, t_opt, n, b):
    return [b * (t > t_opt[j]) for j in range(n)]


def count_events(times, a, b):
    k = 0
    for i in range(len(times)):
        if a < times[i] < b:
            k += 1
    return k


def events_vs_time(c, n, mu, alpha, w, t0, tf, b, d):
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    t_opt = maximize_weighted_activity(b, c, d, t0, tf, alpha, w)

    times_deg, _ = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, c, deg))
    times_prk, _ = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, c, weight))
    times_uniform, _ = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, c, n))
    times_optimal, _ = generate_events(t0, tf, mu, alpha, lambda t: u_opt_inc(t, t_opt, n, b))
    times_unc, _ = generate_events(t0, tf, mu, alpha)

    with open('./results/events_vs_time.pickle', 'wb') as f:
        pickle.dump([times_deg, times_prk, times_uniform, times_optimal, times_unc,
                     c, n, mu, alpha, w, t0, tf, b, d, RND_SEED], f)

    sio.savemat('./results/events_vs_time.mat',
                {'times_deg': times_deg, 'times_prk': times_prk, 'times_uniform': times_uniform, 'times_optimal': times_optimal, 'times_unc': times_unc,
                 'c': c, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED})
    return


def obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d):
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = np.count_nonzero(alpha[i, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((5, len(budget)))
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = maximize_weighted_activity(b, c, d, t0, tf, alpha, w)
        obj[0, i] = eval_weighted_activity(tf, lambda t: mu + u_deg(t, tf, c, deg), d, t0, tf, alpha, w)
        obj[1, i] = eval_weighted_activity(tf, lambda t: mu + u_prk(t, tf, c, weight), d, t0, tf, alpha, w)
        obj[2, i] = eval_weighted_activity(tf, lambda t: mu + u_unf(t, tf, c, n), d, t0, tf, alpha, w)
        obj[3, i] = eval_weighted_activity(tf, lambda t: mu + u_opt_inc(t, t_opt[i, :], n, b), d, t0, tf, alpha, w)
        obj[4, i] = eval_weighted_activity(tf, lambda t: mu, d, t0, tf, alpha, w)

    plt.clf()
    plt.plot(budget, obj[0, :], label="DEG")
    plt.plot(budget, obj[1, :], label="PRK")
    plt.plot(budget, obj[2, :], label="UNF")
    plt.plot(budget, obj[3, :], label="OPT")
    plt.plot(budget, obj[4, :], label="UNC")
    plt.legend(loc="upper left")
    plt.savefig('./results/obj_vs_budget.pdf')

    with open('./results/obj_vs_budget.pickle', 'wb') as f:
        pickle.dump([obj, t_opt, deg, weight,
                     budget, n, mu, alpha, w, t0, tf, b, d, RND_SEED], f)

    sio.savemat('./results/obj_vs_budget.mat',
                {'obj': obj, 't_opt': t_opt, 'deg': deg, 'weight': weight,
                 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED})
    return


def int_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d):
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = np.count_nonzero(alpha[i, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((5, len(budget)))
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)
        obj[0, i] = eval_int_weighted_activity(lambda t: u_deg(t, tf, c, deg), d, t0, tf, alpha, w)
        obj[1, i] = eval_int_weighted_activity(lambda t: mu + u_prk(t, tf, c, weight), d, t0, tf, alpha, w)
        obj[2, i] = eval_int_weighted_activity(lambda t: mu + u_unf(t, tf, c, n), d, t0, tf, alpha, w)
        obj[3, i] = eval_int_weighted_activity(lambda t: mu + u_opt_dec(t, t_opt[i, :], n, b), d, t0, tf, alpha, w)
        obj[4, i] = eval_int_weighted_activity(lambda t: mu, d, t0, tf, alpha, w)

    plt.clf()
    plt.plot(budget, obj[0, :], label="DEG")
    plt.plot(budget, obj[1, :], label="PRK")
    plt.plot(budget, obj[2, :], label="UNF")
    plt.plot(budget, obj[3, :], label="OPT")
    plt.plot(budget, obj[4, :], label="OPT")
    plt.legend(loc="upper left")
    plt.savefig('./results/int_obj_vs_budget.pdf')

    with open('./results/int_obj_vs_budget.pickle', 'wb') as f:
        pickle.dump([obj, t_opt, deg, weight,
                     budget, n, mu, alpha, w, t0, tf, b, d, RND_SEED], f)

    sio.savemat('./results/int_obj_vs_budget.mat',
                {'obj': obj, 't_opt': t_opt, 'deg': deg, 'weight': weight,
                 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED})
    return


def events_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d, itr):
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    event_num = np.zeros([5, len(budget), itr])
    terminal_event_num = np.zeros([5, len(budget), itr])
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = maximize_weighted_activity(b, c, d, t0, tf, alpha, w)

        for j in range(itr):
            times_deg, _ = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, c, deg))
            times_prk, _ = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, c, weight))
            times_uniform, _ = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, c, n))
            times_optimal, _ = generate_events(t0, tf, mu, alpha, lambda t: u_opt_inc(t, t_opt[i, :], n, b))
            times_unc, _ = generate_events(t0, tf, mu, alpha)
            event_num[0, i, j] = len(times_deg)
            event_num[1, i, j] = len(times_prk)
            event_num[2, i, j] = len(times_uniform)
            event_num[3, i, j] = len(times_optimal)
            event_num[4, i, j] = len(times_unc)
            terminal_event_num[0, i, j] = count_events(times_deg, tf-1, tf)
            terminal_event_num[1, i, j] = count_events(times_prk, tf-1, tf)
            terminal_event_num[2, i, j] = count_events(times_uniform, tf-1, tf)
            terminal_event_num[3, i, j] = count_events(times_optimal, tf-1, tf)
            terminal_event_num[4, i, j] = count_events(times_unc, tf - 1, tf)

    event_num_mean = np.mean(event_num, axis=2)
    terminal_event_num_mean = np.mean(terminal_event_num, axis=2)
    # event_num_var = np.var(event_num, axis=2)
    # terminal_event_num_var = np.var(terminal_event_num, axis=2)

    plt.clf()
    plt.plot(budget, terminal_event_num_mean[0, :], marker='.', label="DEG")
    plt.plot(budget, terminal_event_num_mean[1, :], marker='.', label="PRK")
    plt.plot(budget, terminal_event_num_mean[2, :], marker='.', label="UNF")
    plt.plot(budget, terminal_event_num_mean[3, :], marker='.', label="OPT")
    plt.plot(budget, terminal_event_num_mean[4, :], marker='.', label="UNC")
    plt.legend(loc="lower right")
    plt.savefig('./results/max_terminal_events_vs_budget.pdf')

    with open('./results/events_vs_budget.pickle', 'wb') as f:
        pickle.dump([event_num, terminal_event_num, t_opt, deg, weight,
                     budget, n, mu, alpha, w, t0, tf, b, d, itr, RND_SEED], f)

    sio.savemat('./results/events_vs_budget.mat',
                {'event_num': event_num, 'terminal_event_num': terminal_event_num, 't_opt': t_opt, 'deg': deg, 'weight': weight,
                 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED})
    return


def int_events_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d, itr):
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    event_num = np.zeros([5, len(budget), itr])
    terminal_event_num = np.zeros([5, len(budget), itr])
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)

        for j in range(itr):
            times_deg, _ = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, c, deg))
            times_prk, _ = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, c, weight))
            times_uniform, _ = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, c, n))
            times_optimal, _ = generate_events(t0, tf, mu, alpha, lambda t: u_opt_dec(t, t_opt[i, :], n, b))
            times_unc, _ = generate_events(t0, tf, mu, alpha)
            event_num[0, i, j] = len(times_deg)
            event_num[1, i, j] = len(times_prk)
            event_num[2, i, j] = len(times_uniform)
            event_num[3, i, j] = len(times_optimal)
            event_num[4, i, j] = len(times_unc)
            terminal_event_num[0, i, j] = count_events(times_deg, tf - 1, tf)
            terminal_event_num[1, i, j] = count_events(times_prk, tf - 1, tf)
            terminal_event_num[2, i, j] = count_events(times_uniform, tf - 1, tf)
            terminal_event_num[3, i, j] = count_events(times_optimal, tf - 1, tf)
            terminal_event_num[4, i, j] = count_events(times_unc, tf - 1, tf)

    event_num_mean = np.mean(event_num, axis=2)
    # terminal_event_num_mean = np.mean(terminal_event_num, axis=2)
    # event_num_var = np.var(event_num, axis=2)
    # terminal_event_num_var = np.var(terminal_event_num, axis=2)

    plt.clf()
    plt.plot(budget, event_num_mean[0, :], marker='.', label="DEG")
    plt.plot(budget, event_num_mean[1, :], marker='.', label="PRK")
    plt.plot(budget, event_num_mean[2, :], marker='.', label="UNF")
    plt.plot(budget, event_num_mean[3, :], marker='.', label="OPT")
    plt.plot(budget, event_num_mean[4, :], marker='.', label="UNC")
    plt.legend(loc="lower right")
    plt.savefig('./results/max_int_total_events_vs_budget.pdf')

    with open('./results/int_events_vs_budget.pickle', 'wb') as f:
        pickle.dump([event_num, terminal_event_num, t_opt, deg, weight,
                     budget, n, mu, alpha, w, t0, tf, b, d, itr, RND_SEED], f)

    sio.savemat('./results/int_events_vs_budget.mat',
                {'event_num': event_num, 'terminal_event_num': terminal_event_num, 't_opt': t_opt, 'deg': deg, 'weight': weight,
                 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED})
    return


def mehrdad_eval(data_path, itr=30):
    data = sio.loadmat(data_path)
    t0 = data['t0'][0][0]
    tf = data['tf'][0][0]
    n = data['n'][0][0]
    w = data['w'][0][0]
    d = data['d'][:, 0]
    budget = data['budget'][:, 0]
    mu = data['mu'][:, 0]
    alpha = data['alpha']
    lambda_cam = data['lambda_cam']

    obj = np.zeros(budget.shape[0])
    event_num = np.zeros([len(budget), itr])
    terminal_event_num = np.zeros([len(budget), itr])

    for i in range(budget.shape[0]):
        def u_mehrdad(t):
            return [lambda_cam[i, j] for j in range(n)]
        obj[i] = eval_weighted_activity(tf, u_mehrdad, d, t0, tf, alpha, w)
        for j in range(itr):
            times_mehrdad, _ = generate_events(t0, tf, mu, alpha, u_mehrdad)
            event_num[i, j] = len(times_mehrdad)
            terminal_event_num[i, j] = count_events(times_mehrdad, tf - 1, tf)

    with open('./results/events_obj_vs_budget_mehrdad.pickle', 'wb') as f:
        pickle.dump([obj, event_num, terminal_event_num, budget, n, mu, alpha, w, t0, tf, d, itr, RND_SEED], f)

    sio.savemat('./results/events_obj_vs_budget_mehrdad.mat',
                {'obj': obj, 'event_num': event_num, 'terminal_event_num': terminal_event_num,
                 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'd': d, 'seed': RND_SEED})
    return


def shaping_obj_vs_budget(budgets, n, mu, alpha, w, t0, tf, b, ell):
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((4, len(budgets)))
    t_opt = np.zeros((len(budgets), n))
    u_opt = np.zeros((len(budgets), n))
    for i in range(len(budgets)):
        c = budgets[i]
        t_opt[i, :], u_opt[i, :] = maximize_shaping(b, c, ell, t0, tf, alpha, w)
        obj[0, i], unf_int = eval_shaping(np.zeros(n), (deg/sum(deg))*(c/tf)*np.ones(n), ell, tf, alpha, w)
        obj[1, i], unf_int = eval_shaping(np.zeros(n), weight*(c/tf)*np.ones(n), ell, tf, alpha, w)
        obj[2, i], unf_int = eval_shaping(np.zeros(n), (1/n)*(c/tf)*np.ones(n), ell, tf, alpha, w)
        obj[3, i], opt_int = eval_shaping(t_opt[i, :], u_opt[i, :], ell, tf, alpha, w)
        print()

    plt.clf()
    plt.plot(budgets, obj[0, :], label="DEG")
    plt.plot(budgets, obj[1, :], label="PRK")
    plt.plot(budgets, obj[2, :], label="UNF")
    plt.plot(budgets, obj[3, :], label="OPT")
    plt.legend(loc="upper left")
    plt.savefig('./results/shaping_obj_vs_budget.pdf')

    with open('./results/shaping_obj_vs_budget.pickle', 'wb') as f:
        pickle.dump([obj, t_opt, u_opt, budgets, n, mu, alpha, w, t0, tf, b, ell, RND_SEED], f)

    sio.savemat('./results/shaping_obj_vs_budget.mat',
                {'obj': obj, 't_opt': t_opt, 'u_opt': u_opt, 'budget': budgets, 'n': n, 'mu': mu, 'alpha': alpha,
                 'w': w, 't0': t0, 'tf': tf, 'b': b, 'ell': ell, 'seed': RND_SEED})
    return


def main():
    np.random.seed(RND_SEED)
    t0 = 0
    tf = 100
    n = 64
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 1

    b = 100 * mu_max
    c = n * tf * mu_max
    d = np.ones(n)
    budgets = c * np.array([0.005, 0.01, 0.05, 0.1, 1, 5])
    itr = 1

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    ell = 2 * np.array([0.250, 0.250, 0.500, 0.7500] * int(n / 4))
    ell = ell - np.dot(g_int(0, tf, alpha, w), mu)
    print(ell)

    # mehrdad_eval('./data/mehrdad-64.mat')

    # events_vs_time(c*10, n, mu, alpha, w, t0, tf, b, d)
    # obj_vs_budget(budgets, n, mu, alpha, w, t0, tf, b, d)
    # int_obj_vs_budget(budgets, n, mu, alpha, w, t0, tf, b, d)
    # events_vs_budget(budgets, n, mu, alpha, w, t0, tf, b, d, itr)
    # int_events_vs_budget(budgets, n, mu, alpha, w, t0, tf, b, d, itr)

    shaping_obj_vs_budget(budgets, n, mu, alpha, w, t0, tf, b, ell)


if __name__ == '__main__':
    RND_SEED = 1
    main()
