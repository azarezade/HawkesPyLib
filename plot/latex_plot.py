import numpy as np
import matplotlib.pyplot as plt
import pickle


def tex_plot(x, y, path):
    # Set latex params
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)

    # Set fig size, or aspect ratio
    plt.figure(figsize=plt.figaspect(0.8))

    # Plot data
    for row in y:
        plt.plot(x, row, linestyle='-', linewidth=2, marker='o', markerfacecolor='None', markeredgewidth=1.5)

    # Set colors
    plt.gca().set_prop_cycle('color', ['blue', 'orange', 'green', 'crimson', 'purple'])

    # Set label and title
    plt.xlabel(r'$c$ (bubget)', fontsize=20)
    plt.ylabel(r'\vspace{-0.5mm} $\|\mathbb{E}[dN(T)] - \ell \|_2^2$', fontsize=20)

    # Set legend
    plt.legend(['DEG', 'PRK', 'UNF', 'OPL', 'OPT'], loc='upper right', fontsize=16)

    # Set layout
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)

    # Set grid
    plt.grid(color='silver', linestyle='-', linewidth=0.5)
    # plt.grid(True)

    # Set axis properties
    plt.axis('tight')
    # plt.gca().set_xlim(left=-2, right=252)
    # plt.autoscale(enable=True, axis='x', tight=True)

    # Save and show the plot
    plt.savefig(path + '.pdf')
    # plt.show()
    return


if __name__ == '__main__':
    # # Maximization Terminal Objective vs Budget

    # # Maximization Terminal EventsNum vs Budget

    # # Maximization Integral Objective vs Budget

    # # Maximization Integral EventsNum vs Budget

    # # Shaping Terminal Objective vs Budget
    # load data
    name = 'shaping_obj_vs_budget'
    with open('result/' + name + '.pickle', 'rb') as f:
        data = pickle.load(f)
    # obj_mehrdad = np.array([41.0602, 32.6496, 25.4690, 14.2205, 11.8728, 11.5925, 11.5925, 11.5925])
    obj_mehrdad = np.array([41.0602, 32.6496])
    obj = data['obj']
    budget = data['budget']
    obj = np.vstack((obj, obj[-1, :]))
    obj[-2, :] = obj_mehrdad
    # plot and save
    tex_plot(budget, obj, 'result/image/'+name)

    # # # Shaping Terminal EventsNum vs Budget
    # # load data
    # name = 'shaping_events_vs_budget'
    # with open('result/' + name + '.pickle', 'rb') as f:
    #     obj, t_opt, u_opt, budgets, n, mu, alpha, w, t0, tf, b, ell, RND_SEED = pickle.load(f)
    # # plot and save
    # tex_plot(budgets, obj, 'result/image/' + name)

    # # Shaping Integral Objective vs Budget
    # load data
    name = 'shaping_int_obj_vs_budget'
    with open('result/' + name + '.pickle', 'rb') as f:
        data = pickle.load(f)
    # plot and save
    tex_plot(data['budget'], data['obj'], 'result/image/'+name)

    # # Shaping Integral EventsNum vs Budget


