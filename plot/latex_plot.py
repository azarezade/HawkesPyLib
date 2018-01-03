import numpy as np
import matplotlib.pyplot as plt
import pickle
# import os
# PWD = os.path.dirname(os.path.realpath(__file__))


def round10(x, base=10):
    return int(base * round(float(x)/base))


def load(path):
    with open(path + '.pickle', 'rb') as f:
        data = pickle.load(f)
    obj = data['obj']
    budget = data['budget']
    return obj, budget


def tex_plot(x, y, path, xlabel, ylabel, legend, xtick):
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
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    # Set legend
    plt.legend(legend, loc='upper right', fontsize=16)

    # Set layout
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)

    # Set grid
    plt.grid(linestyle='-', linewidth=0.5, alpha=0.5)
    # plt.grid(True)

    # Set x and y axis ticks
    plt.gca().set_xticks(xtick)

    # Set axis properties
    plt.axis('tight')
    # plt.gca().set_xlim(left=-2, right=252)
    # plt.autoscale(enable=True, axis='x', tight=True)

    # Save and show the plot
    plt.savefig(path + '.pdf')
    # plt.show()
    return


def main():
    legend = ['OPT', 'DEG', 'PRK', 'UNF']
    xlabel = r'$c$ (budget)'
    xtick = [0, 50, 100, 150, 200, 250]

    # # Maximization Terminal Objective vs Budget
    # path = '../result/max_obj_vs_budget'
    # obj, budget = load(path)
    # path = '../result/mehrdad_max_obj_vs_budget'
    # obj_mehrdad, budget = load(path)
    # ylabel = r'$w^T E[dN(T)]$'
    # tex_plot(budget, obj, path, xlabel, ylabel, legend, xtick)

    # # Maximization Terminal EventsNum vs Budget
    path = '../result/max_events_vs_budget'
    obj, budget = load(path)
    ylabel = r'$w^T\,\overline{dN(T)}$'
    tex_plot(budget, obj, path, xlabel, ylabel, legend, xtick)

    # Maximization Integral Objective vs Budget
    path = '../result/max_int_obj_vs_budget'
    obj, budget = load(path)
    ylabel = r'$w^T E[\int dN(s)\,ds]$'
    tex_plot(budget, obj, path, xlabel, ylabel, legend, xtick)

    # Maximization Integral EventsNum vs Budget
    path = '../result/max_int_events_vs_budget'
    obj, budget = load(path)
    ylabel = r'$w^T\,\overline{\int dN(s) ds}$'
    tex_plot(budget, obj, path, xlabel, ylabel, legend, xtick)

    # Shaping Terminal Objective vs Budget
    path = '../result/shaping_obj_vs_budget'
    obj, budget = load(path)
    obj_opl = np.array([41.0602, 32.6496, 25.4690, 14.2205, 11.8728, 11.5925, 11.5925, 11.5925])
    obj = np.vstack((obj, obj_opl))
    ylabel = r'$\|E[dN(T)] - \ell \|_2^2$'
    tex_plot(budget, obj, path, xlabel, ylabel, ['OPT', 'DEG', 'PRK', 'UNF', 'OPL'], xtick)

    # Shaping Terminal EventsNum vs Budget
    path = '../result/shaping_events_vs_budget'
    obj, budget = load(path)
    ylabel = r'$\|\overline{dN(T)} - \ell \|_2^2$'
    tex_plot(budget, obj, path, xlabel, ylabel, legend, xtick)

    # Shaping Integral Objective vs Budget
    path = '../result/shaping_int_obj_vs_budget'
    obj, budget = load(path)
    ylabel = r'$\|E[\int dN(s) \, ds] - \ell \|_2^2$'
    tex_plot(budget, obj, path, xlabel, ylabel, legend, xtick)

    # Shaping Integral EventsNum vs Budget
    path = '../result/shaping_int_events_vs_budget'
    obj, budget = load(path)
    ylabel = r'$\|\overline{\int dN(s) ds} - \ell \|_2^2$'
    tex_plot(budget, obj, path, xlabel, ylabel, legend, xtick)
    return


if __name__ == '__main__':
    main()

