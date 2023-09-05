import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import yaml
import argparse

A = 4
E = 0


def xdot_sin(x, t):
    return np.sin(x*np.pi)


def xdot_linear(x, t):
    return -A*x + E*np.sin(x*np.pi)


def load_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='param_exp')
    param = parser.parse_args()
    yaml_file = f'config/{param.yaml}.yaml'
    with open(yaml_file) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args['eps'] = float(args['eps'])
    args['dt'] = float(args['dt'])
    args['dx'] = float(args['dx'])
    return args


def plot_traj(args):
    dx = args['dx']
    n1, n2, n3 = args['n1'], args['n2'], args['n3']
    xlim = (-args['plotlim'], args['plotlim'])

    # x0s = np.random.uniform(-args['xlim'], args['xlim'], n1+n2)  # random full range
    # x0s = np.linspace(args['xlim'] - dx, args['xlim'] + 2*dx, n1+n2)  # full range
    x0s = np.concatenate((np.linspace(dx, args['xlim'] + 2*dx, n1),
                          np.linspace(-args['xlim'] - dx, args['lim1'] - 2*dx, n2)), axis=0)  # in-distribution

    t = np.linspace(0, 1, n3)
    fig, ax = plt.subplots()
    for x0 in x0s:
        x = odeint(xdot_linear, x0, t)
        ax.plot(t, x)
    ax.set_xlim(0, 1)
    ax.set_ylim(xlim[0] - 2*dx, xlim[1] + 3*dx)
    ax.set_aspect(1/(xlim[1] - xlim[0] + 5*dx))
    plt.show()

    
def write_traj(args):
    dx = args['dx']
    n1, n2, n3 = args['n1'], args['n2'], args['n3']
    xlim = (-args['plotlim'], args['plotlim'])

    x0s = np.linspace(xlim[0] - dx, xlim[1] + 2*dx, n1+n2)  # full range
    # x0s = np.concatenate((np.linspace(dx, args['xlim'] + 2*dx, n1),
    #                      np.linspace(-args['xlim'] - dx, args['lim1'] - 2*dx, n2)), axis=0)  # in-distribution
    # x0s = np.concatenate((np.random.uniform(dx, args['xlim'] + 2*dx, n1),
    #                      np.random.uniform(-args['xlim'] - dx, args['lim1'] - 2*dx, n2)), axis=0)  # random in-distribution
    # x0s = np.random.uniform(args['lim1'] - eps, eps, n1+n2)  # random out-of-distribution

    t = np.linspace(0, 1, n3)
    xs = np.zeros((n1+n2, n3))
    for i, x0 in enumerate(x0s):
        x = odeint(xdot_linear, x0, t)
        xs[i] = np.squeeze(x)
    with open(f'data/data_xdlinear_full.txt', 'wb') as f:
        np.savetxt(f, xs)


def main():
    args = load_yaml()
    if args['gentype'] == 'plot':
        plot_traj(args)
    elif args['gentype'] == 'write':
        write_traj(args)

    
if __name__ == '__main__':
    main()
