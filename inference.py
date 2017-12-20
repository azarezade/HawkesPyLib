import numpy as np
import pickle
from scipy.io import loadmat
from tick.inference import HawkesExpKern


def load(matfile):
    data = loadmat(matfile, squeeze_me=True)
    adj = data['Adj']
    events = data['tu_events']
    n = int(adj.shape[0])
    return adj, events, n


def events_to_tick_events(events, n):
    user_times = [[] for i in range(n)]
    for e in events:
        u = int(e[0]) - 1
        t = float(e[1])
        user_times[u].append(t)
    for i in range(len(user_times)):
        user_times[i] = np.array(user_times[i])
    tick_events = [user_times]
    return tick_events


def inference(dataset, decays):
    matfile = "./data/" + dataset + ".mat"
    adj, events, n = load(matfile)

    tick_events = events_to_tick_events(events, n)

    learner = HawkesExpKern(decays=decays, penalty="l1", solver="agd", C=1000, verbose=True)
    learner.fit(tick_events)
    influence_matrix = learner.adjacency
    baseline = learner.baseline
    print("score = {}".format(learner.score()))

    with open("./models/" + dataset + ".pickle", "wb") as f:
        pickle.dump([influence_matrix, baseline, decays, n], f)
    return


if __name__ == "__main__":
    inference("club", 20000)
