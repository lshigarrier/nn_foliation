import matplotlib.pyplot as plt
import pickle
from generate_data import load_yaml


def plot_sigma(args):
    with open(args['sigma_file'], 'rb') as file:
        sigmas = pickle.load(file)
    n = len(sigmas)
    m = n//4 + 1 if n % 4 != 0 else n//4
    fig, axs = plt.subplots(4, m)
    for i in range(n):
        data = sigmas[i]
        i1 = i % 4
        i2 = i//4
        axs[i1, i2].plot(data)
    plt.show()


def plot_save(args):
    fig_file = f'img/{args["prefix"]}_{args["tlim"]}_{args["nb_t"]}_{args["freq"]}_{args["version"]}.fig.pickle'
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    with open(fig_file, 'rb') as file:
        fig = pickle.load(file)
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    ax = fig.get_axes()[0]
    limits = (-args['plotlim'] - args['lim0'], args['plotlim'] + args['lim0'])
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_zlim(*limits)
    plt.close(dummy)
    plt.show()


def main():
    args = load_yaml()
    if args['typeplot'] == 'save':
        plot_save(args)
    elif args['typeplot'] == 'sigma':
        plot_sigma(args)


if __name__ == '__main__':
    main()
