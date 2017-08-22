import pickle
import numpy as np
import networkx as nx
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

from activity_shaping import maximize_weighted_activity, maximize_int_weighted_activity
from evaluations import load_mat, eval_weighted_activity, eval_int_weighted_activity
from event_generation import generate_model, generate_events


def count_events(times, a, b):
    k = 0
    for i in range(len(times)):
        if a < times[i] < b:
            k += 1
    return k


def max_eta_obj_vs_bound(bn, path):
    t0, tf, n, w, mu, alpha, c, d, lambda_cam = load_mat(path)

    deg = np.zeros(n)
    for i in range(n):
        deg[i] = np.count_nonzero(alpha[i, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    def u_mehrdad(t):
        return [lambda_cam[j] for j in range(n)]

    def u_deg(t):
        return (deg / sum(deg)) * (c / tf)

    def u_prk(t):
        return (weight / sum(weight)) * (c / tf)

    def u_uniform(t):
        return [c / (tf * n) for j in range(n)]

    obj_deg = []
    obj_prk = []
    obj_uniform = []
    obj_mehrdad = []
    obj_optimal = []
    for b in range(1, bn, 2):
        t_optimal = maximize_weighted_activity(b*max(lambda_cam), c, d, t0, tf, alpha)

        def u_optimal(t):
            return [b * (t > t_optimal[j]) for j in range(n)]

        obj_optimal.append(eval_weighted_activity(tf, u_optimal, d, t0, tf, alpha, w))

    obj_deg.append(eval_weighted_activity(tf, u_deg, d, t0, tf, alpha, w))
    obj_prk.append(eval_weighted_activity(tf, u_prk, d, t0, tf, alpha, w))
    obj_uniform.append(eval_weighted_activity(tf, u_uniform, d, t0, tf, alpha, w))
    obj_mehrdad.append(eval_weighted_activity(tf, u_mehrdad, d, t0, tf, alpha, w))

    with open('./results/max_obj.pickle', 'wb') as f:
        pickle.dump([obj_deg, obj_prk, obj_uniform, obj_mehrdad, obj_optimal], f)

    x = np.arange(1, bn, 2)
    plt.plot(x, obj_uniform, label="UNF")
    plt.plot(x, obj_deg, label="DEG")
    plt.plot(x, obj_prk, label="PRK")
    plt.plot(x, obj_mehrdad, label="MHD")
    plt.plot(x, obj_optimal, label="OPT")
    plt.legend(loc="upper left")
    plt.savefig('./results/max_eta_obj.pdf')

    return


def max_eta_obj_vs_time(tfs, n, t0, sparsity, mu_max, alpha_max, w, b, d):
    c = tfs[round(len(tfs)/2)] * mu_max

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    obj = np.zeros((4, len(tfs)))
    for i in range(len(tfs)):
        tf = tfs[i]

        deg = np.zeros(n)
        for j in range(n):
            deg[j] = np.count_nonzero(alpha[j, :])

        graph = nx.from_numpy_matrix(alpha)
        pr = nx.pagerank(graph)
        weight = np.asanyarray(list(pr.values()))

        t_optimal = maximize_weighted_activity(b, c, d, t0, tf, alpha)

        def u_deg(t):
            return (deg / sum(deg)) * (c / tf)

        def u_prk(t):
            return (weight / sum(weight)) * (c / tf)

        def u_uniform(t):
            return [c / (tf * n) for j in range(n)]

        def u_optimal(t):
            return [b * (t > t_optimal[j]) for j in range(n)]

        obj[0, i] = eval_weighted_activity(tf, u_deg, d, t0, tf, alpha, w)
        obj[1, i] = eval_weighted_activity(tf, u_prk, d, t0, tf, alpha, w)
        obj[2, i] = eval_weighted_activity(tf, u_uniform, d, t0, tf, alpha, w)
        obj[3, i] = eval_weighted_activity(tf, u_optimal, d, t0, tf, alpha, w)
    np.savetxt('./results/max_eta_obj_vs_time,txt', obj)

    plt.plot(tfs, obj[0, :], label="DEG")
    plt.plot(tfs, obj[1, :], label="PRK")
    plt.plot(tfs, obj[2, :], label="UNF")
    plt.plot(tfs, obj[3, :], label="OPT")
    plt.legend(loc="upper left")
    plt.savefig('./results/max_eta_obj_vs_time.pdf')
    return


def max_eta_event_num_vs_time(tfs, n, t0, sparsity, mu_max, alpha_max, w, b, d):
    c = tfs[round(len(tfs)/2)] * mu_max

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    event_num = np.zeros((4, len(tfs)))
    for i in range(len(tfs)):
        tf = tfs[i]

        deg = np.zeros(n)
        for j in range(n):
            deg[j] = np.count_nonzero(alpha[j, :])

        graph = nx.from_numpy_matrix(alpha)
        pr = nx.pagerank(graph)
        weight = np.asanyarray(list(pr.values()))

        t_optimal = maximize_weighted_activity(b, c, d, t0, tf, alpha)

        def u_deg(t):
            return (deg / sum(deg)) * (c / tf)

        def u_prk(t):
            return (weight / sum(weight)) * (c / tf)

        def u_uniform(t):
            return [c / (tf * n) for j in range(n)]

        def u_optimal(t):
            return [b * (t > t_optimal[j]) for j in range(n)]

        for j in range(50):
            times_deg, _ = generate_events(t0, tf, mu, alpha, u_deg)
            times_prk, _ = generate_events(t0, tf, mu, alpha, u_prk)
            times_uniform, _ = generate_events(t0, tf, mu, alpha, u_uniform)
            times_optimal, _ = generate_events(t0, tf, mu, alpha, u_optimal)

            event_num[0, i] += count_events(times_deg, tf-1, tf)
            event_num[1, i] += count_events(times_prk, tf-1, tf)
            event_num[2, i] += count_events(times_uniform, tf-1, tf)
            event_num[3, i] += count_events(times_optimal, tf-1, tf)

    event_num = event_num / 50
    np.savetxt('./results/max_eta_event_num_vs_time.txt', event_num)

    plt.plot(tfs, event_num[0, :], label="DEG")
    plt.plot(tfs, event_num[1, :], label="PRK")
    plt.plot(tfs, event_num[2, :], label="UNF")
    plt.plot(tfs, event_num[3, :], label="OPT")
    plt.legend(loc="upper left")
    plt.savefig('./results/max_eta_event_num_vs_time.pdf')
    return


def max_eta_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d):
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = np.count_nonzero(alpha[i, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((4, len(budget)))
    for i in range(len(budget)):
        c = budget[i]
        t_optimal = maximize_weighted_activity(b, c, d, t0, tf, alpha, w)

        def u_deg(t):
            return (deg / sum(deg)) * (c / tf)

        def u_prk(t):
            return (weight / sum(weight)) * (c / tf)

        def u_uniform(t):
            return [c / (tf * n) for k in range(n)]

        def u_optimal(t):
            return [b * (t > t_optimal[j]) for j in range(n)]

        obj[0, i] = eval_weighted_activity(tf, u_deg, d, t0, tf, alpha, w)
        obj[1, i] = eval_weighted_activity(tf, u_prk, d, t0, tf, alpha, w)
        obj[2, i] = eval_weighted_activity(tf, u_uniform, d, t0, tf, alpha, w)
        obj[3, i] = eval_weighted_activity(tf, u_optimal, d, t0, tf, alpha, w)
    np.savetxt('./results/max_eta_obj_vs_budget.txt', obj)

    plt.clf()
    plt.plot(budget, obj[0, :], label="DEG")
    plt.plot(budget, obj[1, :], label="PRK")
    plt.plot(budget, obj[2, :], label="UNF")
    plt.plot(budget, obj[3, :], label="OPT")
    plt.legend(loc="upper left")
    plt.savefig('./results/max_eta_obj_vs_budget.pdf')
    return


def max_eta_event_num_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d, itr):
    event_num = np.zeros([4, len(budget), itr])
    terminal_event_num = np.zeros([4, len(budget), itr])
    for i in range(len(budget)):
        c = budget[i]

        deg = np.zeros(n)
        for j in range(n):
            deg[j] = np.count_nonzero(alpha[j, :])

        graph = nx.from_numpy_matrix(alpha)
        pr = nx.pagerank(graph)
        weight = np.asanyarray(list(pr.values()))

        t_optimal = maximize_weighted_activity(b, c, d, t0, tf, alpha, w)

        def u_deg(t):
            return (deg / sum(deg)) * (c / tf)

        def u_prk(t):
            return (weight / sum(weight)) * (c / tf)

        def u_uniform(t):
            return [c / (tf * n) for k in range(n)]

        def u_optimal(t):
            return [b * (t > t_optimal[j]) for j in range(n)]

        for j in range(itr):
            times_deg, _ = generate_events(t0, tf, mu, alpha, u_deg)
            times_prk, _ = generate_events(t0, tf, mu, alpha, u_prk)
            times_uniform, _ = generate_events(t0, tf, mu, alpha, u_uniform)
            times_optimal, _ = generate_events(t0, tf, mu, alpha, u_optimal)

            event_num[0, i, j] = len(times_deg)
            event_num[1, i, j] = len(times_prk)
            event_num[2, i, j] = len(times_uniform)
            event_num[3, i, j] = len(times_optimal)

            terminal_event_num[0, i, j] = count_events(times_deg, tf-1, tf)
            terminal_event_num[1, i, j] = count_events(times_prk, tf-1, tf)
            terminal_event_num[2, i, j] = count_events(times_uniform, tf-1, tf)
            terminal_event_num[3, i, j] = count_events(times_optimal, tf-1, tf)

    # event_num = event_num / itr
    # terminal_event_num = terminal_event_num / itr

    event_num_mean = np.mean(event_num, axis=2)
    event_num_var = np.var(event_num, axis=2)

    terminal_event_num_mean = np.mean(terminal_event_num, axis=2)
    terminal_event_num_var = np.var(terminal_event_num, axis=2)

    np.savetxt('./results/max_eta_event_num_vs_budget_mean.txt', event_num_mean)
    np.savetxt('./results/max_eta_event_num_vs_budget_var.txt', event_num_var)

    np.savetxt('./results/max_eta_terminal_event_num_vs_budget_mean.txt', terminal_event_num_mean)
    np.savetxt('./results/max_eta_terminal_event_num_vs_budget_var.txt', terminal_event_num_var)

    np.save('./results/max_eta_event_num_vs_budget', event_num)
    np.save('./results/max_eta_terminal_event_num_vs_budget', terminal_event_num)

    plt.clf()
    plt.errorbar(budget, event_num_mean[0, :], event_num_var[0, :], marker='.', label="DEG")
    plt.errorbar(budget, event_num_mean[1, :], event_num_var[1, :], marker='.', label="PRK")
    plt.errorbar(budget, event_num_mean[2, :], event_num_var[2, :], marker='.', label="UNF")
    plt.errorbar(budget, event_num_mean[3, :], event_num_var[3, :], marker='.', label="OPT")
    plt.legend(loc="lower right")
    plt.savefig('./results/max_eta_event_num_vs_budget.pdf')

    plt.clf()
    plt.errorbar(budget, terminal_event_num_mean[0, :], terminal_event_num_var[0, :], marker='.', label="DEG")
    plt.errorbar(budget, terminal_event_num_mean[1, :], terminal_event_num_var[1, :], marker='.', label="PRK")
    plt.errorbar(budget, terminal_event_num_mean[2, :], terminal_event_num_var[2, :], marker='.', label="UNF")
    plt.errorbar(budget, terminal_event_num_mean[3, :], terminal_event_num_var[3, :], marker='.', label="OPT")
    plt.legend(loc="lower right")
    plt.savefig('./results/max_eta_terminal_event_num_vs_budget.pdf')
    return


def max_int_eta_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d):
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = np.count_nonzero(alpha[i, :])

    graph = nx.from_numpy_matrix(alpha)
    pr = nx.pagerank(graph)
    weight = np.asanyarray(list(pr.values()))

    obj = np.zeros((4, len(budget)))
    for i in range(len(budget)):
        c = budget[i]
        t_optimal = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)

        def u_deg(t):
            return (deg / sum(deg)) * (c / tf)

        def u_prk(t):
            return (weight / sum(weight)) * (c / tf)

        def u_uniform(t):
            return [c / (tf * n) for k in range(n)]

        def u_optimal(t):
            return [b * (t < t_optimal[j]) for j in range(n)]

        obj[0, i] = eval_int_weighted_activity(u_deg, d, t0, tf, alpha, w)
        obj[1, i] = eval_int_weighted_activity(u_prk, d, t0, tf, alpha, w)
        obj[2, i] = eval_int_weighted_activity(u_uniform, d, t0, tf, alpha, w)
        obj[3, i] = eval_int_weighted_activity(u_optimal, d, t0, tf, alpha, w)
    np.savetxt('./results/max_int_eta_obj_vs_budget.txt', obj)

    plt.clf()
    plt.plot(budget, obj[0, :], label="DEG")
    plt.plot(budget, obj[1, :], label="PRK")
    plt.plot(budget, obj[2, :], label="UNF")
    plt.plot(budget, obj[3, :], label="OPT")
    plt.legend(loc="upper left")
    plt.savefig('./results/max_int_eta_obj_vs_budget.pdf')
    return


def max_int_eta_event_num_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d, itr):
    event_num = np.zeros([4, len(budget), itr])
    terminal_event_num = np.zeros([4, len(budget), itr])
    for i in range(len(budget)):
        c = budget[i]

        deg = np.zeros(n)
        for j in range(n):
            deg[j] = np.count_nonzero(alpha[j, :])

        graph = nx.from_numpy_matrix(alpha)
        pr = nx.pagerank(graph)
        weight = np.asanyarray(list(pr.values()))

        t_optimal = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)

        def u_deg(t):
            return (deg / sum(deg)) * (c / tf)

        def u_prk(t):
            return (weight / sum(weight)) * (c / tf)

        def u_uniform(t):
            return [c / (tf * n) for k in range(n)]

        def u_optimal(t):
            return [b * (t < t_optimal[j]) for j in range(n)]

        for j in range(itr):
            times_deg, _ = generate_events(t0, tf, mu, alpha, u_deg)
            times_prk, _ = generate_events(t0, tf, mu, alpha, u_prk)
            times_uniform, _ = generate_events(t0, tf, mu, alpha, u_uniform)
            times_optimal, _ = generate_events(t0, tf, mu, alpha, u_optimal)

            event_num[0, i, j] = len(times_deg)
            event_num[1, i, j] = len(times_prk)
            event_num[2, i, j] = len(times_uniform)
            event_num[3, i, j] = len(times_optimal)

            terminal_event_num[0, i, j] = count_events(times_deg, tf - 1, tf)
            terminal_event_num[1, i, j] = count_events(times_prk, tf - 1, tf)
            terminal_event_num[2, i, j] = count_events(times_uniform, tf - 1, tf)
            terminal_event_num[3, i, j] = count_events(times_optimal, tf - 1, tf)

    # event_num = event_num / itr
    event_num_mean = np.mean(event_num, axis=2)
    event_num_var = np.var(event_num, axis=2)

    terminal_event_num_mean = np.mean(terminal_event_num, axis=2)
    terminal_event_num_var = np.var(terminal_event_num, axis=2)

    np.savetxt('./results/max_int_eta_event_num_vs_budget_mean.txt', event_num_mean)
    np.savetxt('./results/max_int_eta_event_num_vs_budget_var.txt', event_num_var)

    np.savetxt('./results/max_int_eta_terminal_event_num_vs_budget_mean.txt', terminal_event_num_mean)
    np.savetxt('./results/max_int_eta_terminal_event_num_vs_budget_var.txt', terminal_event_num_var)

    np.save('./results/max_int_eta_event_num_vs_budget', event_num)
    np.save('./results/max_int_eta_terminal_event_num_vs_budget', terminal_event_num)

    plt.clf()
    plt.errorbar(budget, event_num_mean[0, :], event_num_var[0, :], marker='.', label="DEG")
    plt.errorbar(budget, event_num_mean[1, :], event_num_var[1, :], marker='.', label="PRK")
    plt.errorbar(budget, event_num_mean[2, :], event_num_var[2, :], marker='.', label="UNF")
    plt.errorbar(budget, event_num_mean[3, :], event_num_var[3, :], marker='.', label="OPT")
    plt.legend(loc="lower right")
    plt.savefig('./results/max_int_eta_event_num_vs_budget.pdf')

    plt.clf()
    plt.errorbar(budget, terminal_event_num_mean[0, :], terminal_event_num_var[0, :], marker='.', label="DEG")
    plt.errorbar(budget, terminal_event_num_mean[1, :], terminal_event_num_var[1, :], marker='.', label="PRK")
    plt.errorbar(budget, terminal_event_num_mean[2, :], terminal_event_num_var[2, :], marker='.', label="UNF")
    plt.errorbar(budget, terminal_event_num_mean[3, :], terminal_event_num_var[3, :], marker='.', label="OPT")
    plt.legend(loc="lower right")
    plt.savefig('./results/max_int_eta_terminal_event_num_vs_budget.pdf')
    return


def mehrdad_eval(data_path, itr=30):
    data = sio.loadmat(data_path)
    # sio.savemat('./data/pydata-64.mat', {'T': tf, 'N': n, 'w': w, 'mu': mu, 'alpha': alpha, 'c': d, 'C': c / tf})
    # save('mehrdad-64.mat', 't0', 'tf', 'w', 'n', 'mu', 'alpha', 'c', 'd', 'lambda_cam')
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

    np.savetxt('./results/max_eta_event_num_vs_budget_mehrdad_mean.txt', np.mean(event_num, 1))
    np.savetxt('./results/max_eta_event_num_vs_budget_mehrdad_var.txt', np.var(event_num, 1))
    np.savetxt('./results/max_eta_terminal_event_num_vs_budget_mehrdad_mean.txt', np.mean(terminal_event_num, 1))
    np.savetxt('./results/max_eta_terminal_event_num_vs_budget_mehrdad_var.txt', np.mean(terminal_event_num, 1))
    np.savetxt('./results/max_eta_obj_vs_budget_mehrdad.txt', obj)
    return


def main():
    np.random.seed(4)
    t0 = 0
    tf = 100
    n = 16
    sparsity = 0.25
    mu_max = 0.01
    alpha_max = 0.1
    w = 2

    b = 100 * mu_max
    c = 1 * tf * mu_max
    d = np.ones(n)

    # mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)

    mehrdad_eval('./data/mehrdad-64.mat')

    # max_eta_obj_vs_budget([100*c, 200*c, 300*c, 400*c, 500*c], n, mu, alpha, w, t0, tf, b, d)
    # max_eta_event_num_vs_budget([100*c, 200*c, 300*c, 400*c, 500*c], n, mu, alpha, w, t0, tf, b, d, itr=20)

    # max_int_eta_event_num_vs_budget([100*c, 200*c, 300*c, 400*c, 500*c], n, mu, alpha, w, t0, tf, b, d, itr=20)
    # max_int_eta_obj_vs_budget([100*c, 200*c, 300*c, 400*c, 500*c], n, mu, alpha, w, t0, tf, b, d)


if __name__ == '__main__':
    main()
