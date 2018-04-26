import scipy.io as sio
import numpy as np
import pickle
from numpy.linalg import norm
from event_generation import generate_events
from activity_shaping import g_ls_int, activity_shaping_int, eval_activity_shaping_int
# from run_batch_synth import count_user_events
from run_batch_synth import *


if __name__ == '__main__':
    data = sio.loadmat('./data/election.mat')
    adj = data['Adj']
    n = adj.shape[0]
    users = data['tu_events'][:, 0].astype(int) - 1
    times = data['tu_events'][:, 1]

    # load learned model
    with open('./model/election.pickle', 'rb') as f:
        alpha, mu, w, _ = pickle.load(f)

    user_events = count_user_events(times, users, n, 7, 10)
    print(user_events)
    print(sum(user_events))

    # Activity maximization
    t0 = 0
    tf = 3
    b = 100 * max(mu)
    d = np.ones(n)
    budget = np.array([5, 50, 100])  # , 150, 200, 250])
    max_obj_vs_budget(budget, n, mu, alpha, w, t0, tf, b, d)

    # # Activity shaping
    # ell = 4 * np.ones(n)
    # c = sum(ell)
    # b = 10 * max(mu)
    #
    # base_activity = g_ls_int(tf, tf, alpha, w).dot(mu)
    #
    # t_opt, u_opt = activity_shaping_int(b, c, ell, t0, tf, alpha, w)
    # print(min(t_opt), max(t_opt))
    # print(min(u_opt), max(u_opt))
    #
    # obj, _ = eval_activity_shaping_int(t_opt, u_opt, ell-base_activity, tf, alpha, w)
    # print(obj)
    #
    # g = lambda x: np.exp(-w * x)
    # times_opt, users_opt = generate_events(t0, tf, mu, alpha,
    #                                        lambda t: [u_opt[k] * (t < t_opt[k]) for k in range(n)], g=g)
    # print(times_opt)
    # print(users_opt)
    # event_num = count_user_events(times_opt, users_opt, n, t0, tf)
    # obj_emp = norm(event_num - ell) ** 2
    # print(obj_emp)
    # print(event_num)
    # print(ell)
