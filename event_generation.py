# __author__ = 'alza'

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.stats
import statsmodels.api as sm
import matplotlib.pyplot as plt


# TODO: remove default_kernel as input to function
# TODO: use events as objects

def default_kernel(x, w=1):
    return np.exp(-w * x)


def zero_func(x):
    return 0


def generate_model(n, sparsity, mu_max, alpha_max):
    """
    Generate a random sparse matrix as influence matrix of the model, 
    and the random base intensity of the Hawkes process. 

    Args:
        n (int): number of nodes
        sparsity (float): sparsity of the network  
        mu_max (float): max value of the elements in influence matrix alpha 
        alpha_max (float): max value of the elements in the base intensity vector mu

    Returns:
        alpha (ndarray): influence matrix 
        mu (ndarray): base intensity vector
    """
    mu = mu_max * np.random.rand(n,)
    alpha = alpha_max * sp.sparse.rand(n, n, density=sparsity).toarray()
    return mu, alpha


def intensity(t, times, users, mu, alpha, control=zero_func, g=default_kernel, tol=50):
    """
    Intensity of Hawkes process lambda_i(t) = mu + control(t) + sum_{t_j<t} alpha_{ji} g(t-tj)   
    
    Args:
        t (float): time
        times (list): history of time of events up to time t 
        users (list): history of users' index of events up to time t        
        mu (ndarray): base intensity
        alpha (ndarray): influence matrix 
        control: array of control intensity functions (default is zero)
        tol (float): intensity function consider events ti that (t - ti) < 100
        g: kernel function
    
    Returns:
        (ndarray) intensity of the Hawkes process at time t given the history of events
        
    Raises:
        ValueError: in case of invalid user-provided argument.
    """
    m = len(times)
    if m == 0:
        return mu + control(t)
    if times[-1] > t:
        raise ValueError("history times should be lower than current time")
    if len(users) != m:
        raise ValueError("size of times and users should be equal")

    s = np.zeros(mu.shape)
    for i in reversed(range(m)):
        if t - times[i] > tol:
            break
        else:
            s += alpha[:, users[i]] * g(t - times[i])
    return mu + control(t) + s


def plot_intensity(u, t0, tf, times, users, mu, alpha, control=zero_func, g=default_kernel, tol=50):
    x = np.arange(t0, tf, 0.05)
    n = mu.size
    m = x.size
    y = np.zeros([n, m])
    for i in range(m):
        times_less_xi = [t for t in times if t <= x[i]]
        users_less_xi = users[:len(times_less_xi)]
        y[:, i] = intensity(x[i], times_less_xi, users_less_xi, mu, alpha, control, g, tol)

    plt.plot(x, y[u, :])
    plt.show()
    return


def generate_events(t0, tf, mu, alpha, control=zero_func, g=default_kernel, tol=50):
    """
    Sample Hawkes process with mu, alpha using Ogata method from t0 to tf

    Args:
        t0 (float): initial time of simulated events
        tf (float): maximum time of simulated events
        mu (ndarray): base intensity
        alpha (ndarray): influence matrix
        control: array of control intensity functions (default is zero)
        g: kernel function
        tol (float): intensity function consider events ti that (t - ti) < 100

    Returns:
        times (list): times of simulated events
        users (list): users of simulated events
    """
    times = []
    users = []
    t = t0

    while t < tf:
        lambda_m = intensity(t, times, users, mu, alpha, control, g, tol)
        sum_lambda_m = np.sum(lambda_m)

        t += np.random.exponential(1 / sum_lambda_m)
        if t >= tf:
            break

        lambda_t = intensity(t, times, users, mu, alpha, control, g, tol)
        sum_lambda_t = np.sum(lambda_t)

        if np.random.uniform(0, 1) < (sum_lambda_t / sum_lambda_m):
            prob = lambda_t / sum_lambda_t
            u = np.flatnonzero(np.random.multinomial(1, prob, 1))[0]
            times.append(t)
            users.append(u)

        if not (len(times) % 500):
            print("generated {} events up to time {}".format(len(times), times[-1]))

    return times, users


def user_events(u, times, users):
    """
    Return the times of events by user u.
    """
    m = len(times)
    times_u = []
    for i in range(m):
        if users[i] == u:
            times_u.append(times[i])
    return times_u


def verify_events(u, times, users, mu, alpha, w=1):
    """
    Verify that simulated events (times, users) of user u are truly generated from 
     Hawkes process with (mu, alpha) by plotting qqplot. 
    It returns the array of integrals, \int_{t_{i-1}}^{t_i} \lambda(s) ds 
    """
    times_u = user_events(u, times, users)
    n = mu.size
    m = len(times_u)
    lambda_integrals = np.zeros(m)
    for k in range(1, m):
        lambda_integrals[k] = mu[u] * (times_u[k] - times_u[k - 1])
        for v in range(n):
            times_v = user_events(v, times, users)
            for l in range(len(times_v)):
                if not (times_v[l] < times_u[k - 1]):
                    break
                else:
                    lambda_integrals[k] += (alpha[u, v] / w) * (np.exp(-w * (times_u[k - 1] - times_v[l])) -
                                                                np.exp(-w * (times_u[k] - times_v[l])))
            for p in range(l, len(times_v)):
                if times_u[k - 1] < times_v[l] < times_u[k]:
                    lambda_integrals[k] += (alpha[u, v] / w) * (1 - np.exp(-w * (times_u[k] - times_v[l])))

    sm.qqplot(lambda_integrals, sp.stats.expon, loc=0, scale=1, line='45')
    plt.show()
    return


def main():
    np.random.seed(100)

    t0 = 0
    tf = 100000
    n = 5
    sparsity = 0.3
    mu_max = 0.01
    alpha_max = 0.1

    mu, alpha = generate_model(n, sparsity, mu_max, alpha_max)
    times, users = generate_events(t0, tf, mu, alpha)

    print(len(times))
    print(times)


if __name__ == '__main__':
    main()
