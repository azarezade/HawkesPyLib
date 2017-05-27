# __author__ = 'Ali_Zarezade'


from event_generation import *
from activity_shaping import *


def load_mat_file(path):
    mat_contents = sio.loadmat(path)
    t0 = float(mat_contents['t0'][0][0])
    tf = float(mat_contents['tf'][0][0])
    n = int(mat_contents['n'][0][0])
    sparsity = float(mat_contents['sparsity'][0][0])
    mu_max = float(mat_contents['mu_max'][0][0])
    alpha_max = float(mat_contents['alpha_max'][0][0])
    mu = mat_contents['mu'][:, 0]
    alpha = mat_contents['alpha']
    b = float(mat_contents['b'][0][0])
    c = float(mat_contents['c'][0][0])
    d = mat_contents['d'][:, 0]
    lambda_cam = mat_contents['lambda_cam'][:, 0]
    return t0, tf, n, sparsity, mu_max, alpha_max, mu, alpha, b, c, d, lambda_cam


def compare_weighted_activity(mat_address):
    t0, tf, n, sparsity, mu_max, alpha_max, mu, alpha, b, c, d, u_cam = load_mat_file(mat_address)
    t_star = maximize_weighted_activity(b, c, d, t0, tf, alpha)

    def u_optimal(t):
        u = np.zeros(n)
        for i in range(n):
            u[i] = b * (t > t_star[i])
        return u

    times_base, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha)
    times_mehrdad, _ = generate_events(t0=t0, tf=tf, mu=mu + u_cam, alpha=alpha)
    times_optimal, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_optimal)

    print("budget={}, \t base intensity sum".format(c, tf*sum(mu)))
    print("base \t\t num of event={}".format(len(times_base)))
    print("mehrdad \t num of event={} \t increase(%)={}".
          format(len(times_mehrdad), 100 * (len(times_mehrdad)-len(times_base)) / len(times_base)))
    print("optimal \t num of event={} \t increase(%)={}".
          format(len(times_optimal), 100 * (len(times_optimal) - len(times_base)) / len(times_base)))
    return


def compare_int_weighted_activity(t0, tf, b, c, d, w, n, sparsity, mu_max, alpha_max):
    """ Compare the proposed method with poisson baseline. """
    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    t_star = maximize_int_weighted_activity(b, c, d, t0, tf, alpha, w)

    def u_optimal(t):
        return [b * (t < t_star[i]) for i in range(n)]

    def u_poisson(t):
        return [c / (tf * n) for i in range(n)]

    obj_poisson = eval_int_weighted_activity(u_poisson, d, t0, tf, alpha, w)
    obj_optimal = eval_int_weighted_activity(u_optimal, d, t0, tf, alpha, w)
    print("obj_poisson={} \t\t ".format(obj_poisson))
    print("obj_optimal={} \t\t ".format(obj_optimal))

    times_base, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha)
    times_poisson, _ = generate_events(t0=t0, tf=tf, mu=mu + (c / (tf * n)), alpha=alpha)
    times_optimal, _ = generate_events(t0=t0, tf=tf, mu=mu, alpha=alpha, control=u_optimal)

    print("poisson \t num of event={} \t increase(%)={}".
          format(len(times_poisson), 100 * (len(times_poisson) - len(times_base)) / len(times_base)))
    print("optimal \t num of event={} \t increase(%)={}".
          format(len(times_optimal), 100 * (len(times_optimal) - len(times_base)) / len(times_base)))
    return


if __name__ == '__main__':
    # np.random.seed(10)
    t0 = 0
    tf = 200
    n = 10
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1
    w = 1

    b = 1. * mu_max
    c = 1. * tf * mu_max
    d = np.ones(n)

    compare_weighted_activity('./data/mehrdad_shaping.mat')
    #
    # compare_int_weighted_activity(t0, tf, b, c, d, w, n, sparsity, mu_max, alpha_max)
