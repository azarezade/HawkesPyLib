# __author__ = 'Ali_Zarezade'

# # Set limit to the number of cores:
# import mkl
# mkl.set_num_threads(n)

import numpy as np
import scipy.io as sio
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from numpy.linalg import inv, norm
from activity_maximization import activity_max, activity_max_int, eval_activity_max, eval_activity_max_int
from event_generation import generate_model, generate_events
from activity_shaping import eval_activity_shaping, activity_shaping, g_ls_int, g_max_int, activity_shaping_int, eval_activity_shaping_int


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


def count_user_events(times, users, n, a, b):
    count = np.zeros(n)
    for i in range(len(times)):
        if a < times[i] < b:
            count[users[i]] += 1
    return count


def mehrdad_max_events_and_obj_vs_budget(data_path, itr=30):
    g = lambda x: np.exp(-w * x)
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
        obj[i] = eval_activity_max(tf, u_mehrdad, d, t0, tf, alpha, w)
        for j in range(itr):
            times_mehrdad, _ = generate_events(t0, tf, mu, alpha, u_mehrdad, g=g)
            event_num[i, j] = len(times_mehrdad)
            terminal_event_num[i, j] = count_events(times_mehrdad, tf - 1, tf)

    data = {'obj': obj, 'event_num': event_num, 'terminal_event_num': terminal_event_num, 'budget': budget, 'n': n,
            'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'd': d, 'seed': RND_SEED}
    sio.savemat('./result/mehrdad_max_events_and_obj_vs_budget.mat', data)
    with open('./result/mehrdad_max_events_and_obj_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def max_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d):
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = np.count_nonzero(alpha[i, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((4, len(budget)))
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = activity_max(b, c, d, t0, tf, alpha, w)
        obj[0, i] = eval_activity_max(tf, lambda t: mu + u_opt_inc(t, t_opt[i, :], n, b), d, t0, tf, alpha, w)
        obj[1, i] = eval_activity_max(tf, lambda t: mu + u_deg(t, tf, c, deg), d, t0, tf, alpha, w)
        obj[2, i] = eval_activity_max(tf, lambda t: mu + u_prk(t, tf, c, weight), d, t0, tf, alpha, w)
        obj[3, i] = eval_activity_max(tf, lambda t: mu + u_unf(t, tf, c, n), d, t0, tf, alpha, w)
        # obj[4, i] = eval_activity_max(tf, lambda t: mu, d, t0, tf, alpha, w)

    data = {'obj': obj, 't_opt': t_opt, 'deg': deg, 'weight': weight, 'budget': budget, 'n': n, 'mu': mu,
            'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED}
    sio.savemat('./result/max_obj_vs_budget.mat', data)
    with open('./result/max_obj_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def max_events_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d, itr):
    g = lambda x: np.exp(-w * x)
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    event_num = np.zeros([4, len(budget), itr])
    terminal_event_num = np.zeros([4, len(budget), itr])
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = activity_max(b, c, d, t0, tf, alpha, w)

        for j in range(itr):
            times_optimal, _ = generate_events(t0, tf, mu, alpha, lambda t: u_opt_inc(t, t_opt[i, :], n, b), g=g)
            times_deg, _ = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, c, deg), g=g)
            times_prk, _ = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, c, weight), g=g)
            times_uniform, _ = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, c, n), g=g)
            # times_unc, _ = generate_events(t0, tf, mu, alpha, g=g)
            event_num[0, i, j] = len(times_optimal)
            event_num[1, i, j] = len(times_deg)
            event_num[2, i, j] = len(times_prk)
            event_num[3, i, j] = len(times_uniform)
            # event_num[4, i, j] = len(times_unc)
            terminal_event_num[0, i, j] = count_events(times_optimal, tf-1, tf)
            terminal_event_num[1, i, j] = count_events(times_deg, tf-1, tf)
            terminal_event_num[2, i, j] = count_events(times_prk, tf-1, tf)
            terminal_event_num[3, i, j] = count_events(times_uniform, tf-1, tf)
            # terminal_event_num[4, i, j] = count_events(times_unc, tf - 1, tf)
    obj = np.mean(terminal_event_num, 2)

    data = {'obj': obj, 'event_num': event_num, 'terminal_event_num': terminal_event_num, 't_opt': t_opt, 'deg': deg,
            'weight': weight, 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b,
            'd': d, 'seed': RND_SEED}
    sio.savemat('./result/max_events_vs_budget.mat', data)
    with open('./result/max_events_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def max_int_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d):
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = np.count_nonzero(alpha[i, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((4, len(budget)))
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = activity_max_int(b, c, d, t0, tf, alpha, w)
        obj[0, i] = eval_activity_max_int(lambda t: mu + u_opt_dec(t, t_opt[i, :], n, b), d, t0, tf, alpha, w)
        obj[1, i] = eval_activity_max_int(lambda t: u_deg(t, tf, c, deg), d, t0, tf, alpha, w)
        obj[2, i] = eval_activity_max_int(lambda t: mu + u_prk(t, tf, c, weight), d, t0, tf, alpha, w)
        obj[3, i] = eval_activity_max_int(lambda t: mu + u_unf(t, tf, c, n), d, t0, tf, alpha, w)
        # obj[4, i] = eval_activity_max_int(lambda t: mu, d, t0, tf, alpha, w)

    data = {'obj': obj, 't_opt': t_opt, 'deg': deg, 'weight': weight, 'budget': budget, 'n': n, 'mu': mu,
            'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED}
    sio.savemat('./result/max_int_obj_vs_budget.mat', data)
    with open('./result/max_int_obj_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def max_int_events_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d, itr):
    g = lambda x: np.exp(-w * x)
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    event_num = np.zeros([4, len(budget), itr])
    terminal_event_num = np.zeros([4, len(budget), itr])
    t_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :] = activity_max_int(b, c, d, t0, tf, alpha, w)

        for j in range(itr):
            times_optimal, _ = generate_events(t0, tf, mu, alpha, lambda t: u_opt_dec(t, t_opt[i, :], n, b), g=g)
            times_deg, _ = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, c, deg), g=g)
            times_prk, _ = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, c, weight), g=g)
            times_uniform, _ = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, c, n), g=g)
            # times_unc, _ = generate_events(t0, tf, mu, alpha, g=g)
            event_num[0, i, j] = len(times_optimal)
            event_num[1, i, j] = len(times_deg)
            event_num[2, i, j] = len(times_prk)
            event_num[3, i, j] = len(times_uniform)
            # event_num[4, i, j] = len(times_unc)
            terminal_event_num[0, i, j] = count_events(times_optimal, tf - 1, tf)
            terminal_event_num[1, i, j] = count_events(times_deg, tf - 1, tf)
            terminal_event_num[2, i, j] = count_events(times_prk, tf - 1, tf)
            terminal_event_num[3, i, j] = count_events(times_uniform, tf - 1, tf)
            # terminal_event_num[4, i, j] = count_events(times_unc, tf - 1, tf)

    obj = np.mean(event_num, 2)
    data = {'obj': obj, 'event_num': event_num, 'terminal_event_num': terminal_event_num, 't_opt': t_opt, 'deg': deg,
            'weight': weight, 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b,
            'd': d, 'seed': RND_SEED}
    sio.savemat('./result/max_int_events_vs_budget.mat', data)
    with open('./result/max_int_events_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def shaping_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, ell):
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((4, len(budget)))
    t_opt = np.zeros((len(budget), n))
    u_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :], u_opt[i, :] = activity_shaping(b, c, ell, t0, tf, alpha, w)
        obj[0, i], _ = eval_activity_shaping(t_opt[i, :], u_opt[i, :], ell, tf, alpha, w)
        obj[1, i], _ = eval_activity_shaping(np.zeros(n), (deg / sum(deg)) * (c / tf) * np.ones(n), ell, tf, alpha, w)
        obj[2, i], _ = eval_activity_shaping(np.zeros(n), weight * (c / tf) * np.ones(n), ell, tf, alpha, w)
        obj[3, i], _ = eval_activity_shaping(np.zeros(n), (1 / n) * (c / tf) * np.ones(n), ell, tf, alpha, w)
        print("used_budget={}, total_budget={}, obj={}".format(np.dot(tf-t_opt[i, :], u_opt[i, :]), c, obj[:,i]))

    data = {'obj': obj, 't_opt': t_opt, 'u_opt': u_opt, 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w,
            't0': t0, 'tf': tf, 'b': b, 'ell': ell, 'seed': RND_SEED}
    sio.savemat('./result/shaping_obj_vs_budget.mat', data)
    with open('./result/shaping_obj_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def shaping_events_vs_budget(budget, n, mu, alpha, w, t0, tf, b, ell, itr):
    g = lambda x: np.exp(-w * x)
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    terminal_event_num = np.zeros([4, len(budget), n])
    t_opt = np.zeros((len(budget), n))
    u_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :], u_opt[i, :] = activity_shaping(b, c, ell, t0, tf, alpha, w)

        for j in range(itr):
            times_opt, users_opt = generate_events(t0, tf, mu, alpha, lambda t: [u_opt[i,k] * (t > t_opt[i,k]) for k in range(n)], g=g)
            times_deg, users_deg = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, c, deg), g=g)
            times_prk, users_prk = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, c, weight), g=g)
            times_unf, users_unf = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, c, n), g=g)
            # times_unc, users_unc = generate_events(t0, tf, mu, alpha, g=g)
            terminal_event_num[0, i, :] += count_user_events(times_opt, users_opt, n, tf-1, tf)
            terminal_event_num[1, i, :] += count_user_events(times_deg, users_deg, n, tf-1, tf)
            terminal_event_num[2, i, :] += count_user_events(times_prk, users_prk, n, tf-1, tf)
            terminal_event_num[3, i, :] += count_user_events(times_unf, users_unf, n, tf-1, tf)
            # terminal_event_num[4, i, :] += count_user_events(times_unc, users_unc, n, tf-1, tf)
        terminal_event_num[:, i, :] = terminal_event_num[:, i, :] / itr

    obj = np.zeros((4, len(budget)))
    for i in range(4):
        for j in range(len(budget)):
            obj[i, j] = norm(terminal_event_num[i, j, :] - ell) ** 2

    data = {'terminal_event_num': terminal_event_num, 'obj': obj, 't_opt': t_opt, 'deg': deg, 'weight': weight,
            'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'ell': ell,
            'seed': RND_SEED}
    sio.savemat('./result/shaping_events_vs_budget.mat', data)
    with open('./result/shaping_events_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def shaping_int_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, ell):
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((4, len(budget)))
    t_opt = np.zeros((len(budget), n))
    u_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :], u_opt[i, :] = activity_shaping_int(b, c, ell, t0, tf, alpha, w)
        obj[0, i], _ = eval_activity_shaping_int(t_opt[i, :], u_opt[i, :], ell, tf, alpha, w)
        obj[1, i], _ = eval_activity_shaping_int(tf * np.ones(n), (deg / sum(deg)) * (c / tf) * np.ones(n), ell, tf, alpha, w)
        obj[2, i], _ = eval_activity_shaping_int(tf * np.ones(n), weight * (c / tf) * np.ones(n), ell, tf, alpha, w)
        obj[3, i], _ = eval_activity_shaping_int(tf * np.ones(n), (1 / n) * (c / tf) * np.ones(n), ell, tf, alpha, w)
        print("used_budget={}, total_budget={}, obj={}".format(np.dot(t_opt[i, :], u_opt[i, :]), c, obj[:,i]))

    data = {'obj': obj, 't_opt': t_opt, 'u_opt': u_opt, 'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w,
            't0': t0, 'tf': tf, 'b': b, 'ell': ell, 'seed': RND_SEED}
    sio.savemat('./result/shaping_int_obj_vs_budget.mat', data)
    with open('./result/shaping_int_obj_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def shaping_int_events_vs_budget(budget, n, mu, alpha, w, t0, tf, b, ell, base_activity, itr):
    g = lambda x: np.exp(-w*x)
    deg = np.zeros(n)
    for k in range(n):
        deg[k] = np.count_nonzero(alpha[k, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    event_num = np.zeros([4, len(budget), n])
    t_opt = np.zeros((len(budget), n))
    u_opt = np.zeros((len(budget), n))
    for i in range(len(budget)):
        c = budget[i]
        t_opt[i, :], u_opt[i, :] = activity_shaping_int(b, c, ell - base_activity, t0, tf, alpha, w)
        # c_opt = np.dot(t_opt[i, :], u_opt[i, :])

        for k in range(itr):
            times_opt, users_opt = generate_events(t0, tf, mu, alpha, lambda t: [u_opt[i, k] * (t < t_opt[i, k]) for k in range(n)], g=g)
            times_deg, users_deg = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, c, deg), g=g)
            times_prk, users_prk = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, c, weight), g=g)
            times_unf, users_unf = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, c, n), g=g)
            # times_unc, users_unc = generate_events(t0, tf, mu, alpha, g=g)

            event_num[0, i, :] += count_user_events(times_opt, users_opt, n, 0, tf)
            event_num[1, i, :] += count_user_events(times_deg, users_deg, n, 0, tf)
            event_num[2, i, :] += count_user_events(times_prk, users_prk, n, 0, tf)
            event_num[3, i, :] += count_user_events(times_unf, users_unf, n, 0, tf)
            # event_num[4, i, :] += count_user_events(times_unc, users_unc, n, 0, tf)
        event_num[:, i, :] = event_num[:, i, :] / itr

    obj = np.zeros((4, len(budget)))
    for i in range(4):
        for j in range(len(budget)):
            obj[i, j] = norm(event_num[i, j, :] - ell)**2

    data = {'event_num': event_num, 'obj': obj, 't_opt': t_opt, 'u_opt': u_opt, 'deg': deg, 'weight': weight,
            'budget': budget, 'n': n, 'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'ell': ell,
            'base_activity': base_activity, 'seed': RND_SEED}
    sio.savemat('./result/shaping_int_events_vs_budget.mat', data)
    with open('./result/shaping_int_events_vs_budget.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def max_int_events_vs_time(budget, n, mu, alpha, w, t0, tf, b, d, itr):
    g = lambda x: np.exp(-w * x)
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    times = np.zeros([5, itr], dtype=object)
    users = np.zeros([5, itr], dtype=object)
    t_opt = activity_max_int(b, budget, d, t0, tf, alpha, w)
    for i in range(itr):
        times[0, i], users[0, i] = generate_events(t0, tf, mu, alpha, lambda t: u_opt_dec(t, t_opt, n, b), g=g)
        times[1, i], users[1, i] = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, budget, deg), g=g)
        times[2, i], users[2, i] = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, budget, weight), g=g)
        times[3, i], users[3, i] = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, budget, n), g=g)
        times[4, i], users[4, i] = generate_events(t0, tf, mu, alpha)

    data = {'t_opt': t_opt, 'times': times, 'users': users, 'deg': deg, 'weight': weight, 'budget': budget, 'n': n,
            'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'd': d, 'seed': RND_SEED}
    sio.savemat('./result/max_int_events_vs_time.mat', data)
    with open('./result/max_int_events_vs_time.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def shaping_int_events_vs_time(budget, n, mu, alpha, w, t0, tf, b, ell, base_activity, itr):
    g = lambda x: np.exp(-w * x)
    deg = np.zeros(n)
    for j in range(n):
        deg[j] = np.count_nonzero(alpha[j, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    times = np.zeros([5, itr], dtype=object)
    users = np.zeros([5, itr], dtype=object)
    t_opt, u_opt = activity_shaping_int(b, budget, ell - base_activity, t0, tf, alpha, w)
    for i in range(itr):
        times[0, i], users[0, i] = generate_events(t0, tf, mu, alpha, lambda t: [u_opt[k] * (t < t_opt[k]) for k in range(n)], g=g)
        times[1, i], users[1, i] = generate_events(t0, tf, mu, alpha, lambda t: u_deg(t, tf, budget, deg), g=g)
        times[2, i], users[2, i] = generate_events(t0, tf, mu, alpha, lambda t: u_prk(t, tf, budget, weight), g=g)
        times[3, i], users[3, i] = generate_events(t0, tf, mu, alpha, lambda t: u_unf(t, tf, budget, n), g=g)
        times[4, i], users[4, i] = generate_events(t0, tf, mu, alpha)
    data = {'t_opt': t_opt, 'times': times, 'users': users, 'deg': deg, 'weight': weight, 'budget': budget, 'n': n,
            'mu': mu, 'alpha': alpha, 'w': w, 't0': t0, 'tf': tf, 'b': b, 'ell': ell, 'base_activity': base_activity, 
            'seed': RND_SEED}
    sio.savemat('./result/shaping_int_events_vs_time.mat', data)
    with open('./result/shaping_int_events_vs_time.pickle', 'wb') as f:
        pickle.dump(data, f)
    return


def main():
    np.random.seed(RND_SEED)
    t0 = 0
    tf = 100
    n = 64
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w_m = 1
    w_s = 2
    b = 100 * mu_max
    d = np.ones(n)
    budget = np.array([0.5, 10, 20, 50, 100, 150, 200, 250])
    itr = 20

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    # ell for shaping terminal
    ell = 2 * np.array([0.250, 0.250, 0.500, 0.7500] * int(n / 4))
    ell = ell - np.dot(g_max_int(0, tf, alpha, w_s), mu)
    # ell for shaping integral
    ell_int = 6 * np.array([0.250, 0.250, 0.500, 0.7500] * int(n / 4))
    base_activity = g_ls_int(tf, tf, alpha, w_s).dot(mu)
    ell_int = ell_int + base_activity

    # mehrdad_max_events_and_obj_vs_budget('./data/mehrdad-64.mat')

    # max_obj_vs_budget(budget, n, mu, alpha, w_m, t0, tf, b, d)
    # max_events_vs_budget(budget, n, mu, alpha, w_m, t0, tf, b, d, itr)
    # max_int_obj_vs_budget(budget, n, mu, alpha, w_m, t0, tf, b, d)
    # max_int_events_vs_budget(budget, n, mu, alpha, w_m, t0, tf, b, d, itr)

    # shaping_obj_vs_budget(budget, n, mu, alpha, w_s, t0, tf, b, ell)
    # shaping_events_vs_budget(budget, n, mu, alpha, w_s, t0, tf, b, ell, itr)
    # shaping_int_obj_vs_budget(budget, n, mu, alpha, w_s, t0, tf, b, ell_int)
    # shaping_int_events_vs_budget(budget, n, mu, alpha, w_s, t0, tf, b, ell_int, base_activity, itr)

    # max_int_events_vs_time(budget[-1], n, mu, alpha, w_m, t0, tf, b, d, itr)
    # shaping_int_events_vs_time(budget[-1], n, mu, alpha, w_s, t0, tf, b, ell, base_activity, itr)


if __name__ == '__main__':
    RND_SEED = 4
    main()
