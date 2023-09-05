from distribution import Distribution
from generate_data import xdot_sin, xdot_linear, load_yaml
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pickle


def ortho(u, v):
    """
    Euclidean orthonormal basis
    """
    new_u = u/np.linalg.norm(u)
    new_v = v - np.dot(v.T, new_u)*new_u
    return new_u, new_v/np.linalg.norm(new_v)


def proj(p, u, v):
    """
    Orthogonal projection with Euclidean metric and normalization
    """
    u, v = ortho(u, v)
    mat = np.concatenate((np.expand_dims(u, axis=1), np.expand_dims(v, axis=1)), axis=1)
    inv_mat = np.linalg.inv(np.dot(mat.T, mat))
    mat = np.dot(mat, np.dot(inv_mat, mat.T))
    new_p = np.dot(mat, p)
    return new_p/np.linalg.norm(new_p)


def rotate(p, u, v, angle):
    """
    Rotation with Euclidean metric
    """
    rot_matrix = get_rotation_matrix(u, v, inv=False, angle=angle)
    return np.dot(rot_matrix, p)


def get_rotation_matrix(u, v, inv=False, angle=None):

    n = u.shape[0]
    if angle is None:
        angle = - np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    if inv:
        angle = - angle
    u, v = ortho(u, v)
    u = np.expand_dims(u, axis=1)
    v = np.expand_dims(v, axis=1)
    return np.eye(n) + np.sin(angle) * (np.dot(v, u.T) - np.dot(u, v.T)) + (np.cos(angle) - 1) * (np.dot(u, u.T) + np.dot(v, v.T))


def get_basis(x, distri, kernel=False, acc=False):
    
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(-1)
    if acc:
        vect, pred = distri.compute_kernel(x.float(), kernel, acc)
        vect = vect[0].detach().cpu().numpy()
        return vect, pred
    vect = distri.compute_kernel(x.float(), kernel, acc)
    vect = vect[0].detach().cpu().numpy()
    return vect


def x_dot(x, t, distri, v0, kernel=False, acc=None):

    if acc is not None:
        basis, pred = get_basis(x, distri, kernel, acc=True)
        acc.append(pred[0, 1])
    else:
        basis = get_basis(x, distri, kernel)
    if kernel:
        return basis / np.linalg.norm(basis)
    return proj(v0, basis[:, 0], basis[:, 1])  # using the Euclidean parallel transport


def initialize():

    print("Initialization ... ", end='')
    distri = Distribution()
    print("completed!")
    return distri


def euler(deriv, x0, time_range, reverse=False, args=()):
    x = [x0]
    t0 = time_range[0]
    dx = deriv(x0, t0, *args)
    if reverse:
        dx = -dx
    for i in range(1, len(time_range)):
        dt = time_range[i] - t0
        new_dx = deriv(x[-1], t0, *args)
        if np.dot(dx, new_dx) < 0:
            new_dx = - new_dx[:]
        dx = new_dx[:]
        x.append(x[-1] + dt * dx)
        t0 = time_range[i]
    return np.array(x)


def rungekutta4(f, y0, t, reverse=False, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    k1 = f(y[0], t[0], *args)
    if reverse:
        k1 = -k1
    for i in range(n - 1):
        h = t[i+1] - t[i]
        new_k1 = f(y[i], t[i], *args)
        if np.dot(k1, new_k1) < 0:
            new_k1 = - new_k1
        k1 = new_k1
        new_k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        if np.dot(k1, new_k2) < 0:
            new_k2 = - new_k2
        k2 = new_k2
        new_k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        if np.dot(k1, new_k3) < 0:
            new_k3 = - new_k3
        k3 = new_k3
        new_k4 = f(y[i] + k3 * h, t[i] + h, *args)
        if np.dot(k1, new_k4) < 0:
            new_k4 = - new_k4
        k4 = new_k4
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y


def plot_leaves(args):
    tic = time.time()
    distri = initialize()
    u = np.array([1, 0, 0])
    v = np.array([1, 1, 1])
    rot_matrix = get_rotation_matrix(u, v)
    tlim = args['tlim']
    nb_t = args['nb_t']
    freq = args['freq']
    version = args['version']
    dt = args['dt']
    dx = 1.1*args['dx']
    xlim = args['xlim']
    plotlim = args['plotlim']
    lim0 = args['lim0']
    lim1 = args['lim1']
    lim2 = args['lim2']
    rotation = args['rotation']
    fig_file = f'kernel_sin_{tlim}_{nb_t}_{freq}_{version}'
    time_range = [0, dt, 2*dt]
    time_array = np.linspace(0, tlim, nb_t)
    if args['color']:
        x0s = [np.arange(-xlim - lim0 - dx, -xlim, dx),
               np.arange(-xlim - dx, lim1, dx),
               np.arange(lim1 - dx, lim2, dx),
               np.arange(lim2 - dx, xlim, dx),
               np.arange(xlim - dx, xlim + lim0, dx)]
        colors = ['b', 'r', 'b', 'r', 'b']
    else:
        x0s = [np.arange(-xlim - lim0 - dx, xlim + lim0, dx)]
        colors = ['r']
    if args['system'] == 'sin':
        xdot_system = xdot_sin
    elif args['system'] == 'linear':
        xdot_system = xdot_linear
    ntot = sum(x0s[i].shape[0] for i in range(len(x0s)))
    xembs = [np.zeros((x0s[i].shape[0], 3)) for i in range(len(x0s))]
    xker = np.zeros((ntot, 3))
    leaves = []

    k = 0
    sigmas = []
    for i in range(len(x0s)):
        for j in range(x0s[i].shape[0]):
            x0 = np.array(odeint(xdot_system, x0s[i][j], time_range)).squeeze()
            xembs[i][j, :] = np.dot(rot_matrix, x0)/np.sqrt(3) if rotation else x0
            if k % freq == 1:
                basis = get_basis(x0, distri, kernel=True)
                basis = basis / np.linalg.norm(basis)
                if args['acc']:
                    acc = []
                else:
                    acc = None
                arguments = (distri, None, True, acc)
                xl = rungekutta4(x_dot, x0, time_array, args=arguments)
                if args['acc']:
                    sigmas.append(acc.copy())
                    acc = []
                else:
                    acc = None
                arguments = (distri, None, True, acc)
                xr = rungekutta4(x_dot, x0, time_array, reverse=True, args=arguments)
                if args['acc']:
                    sigmas.append(acc.copy())
                if rotation:
                    xl = np.dot(rot_matrix, xl.T).T / np.sqrt(3)
                    xr = np.dot(rot_matrix, xr.T).T / np.sqrt(3)
                leaves.append(xl)
                leaves.append(xr)
                xker[k, :] = np.dot(rot_matrix, basis)/np.sqrt(3) if rotation else basis
            if k % 10 == 1:
                print(f'{k/ntot*100:.2f}%', end=' ')
            k += 1

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for i in range(len(xembs)):
        ax.plot(xembs[i][:, 0], xembs[i][:, 1], xembs[i][:, 2], color=colors[i])
    for leaf in leaves:
        ax.plot(leaf[:, 0], leaf[:, 1], leaf[:, 2], color='g', alpha=0.5)
    xemb = np.concatenate(xembs, axis=0)
    ax.quiver(xemb[:, 0], xemb[:, 1], xemb[:, 2], xker[:, 0], xker[:, 1], xker[:, 2], length=1, color='m')

    ax.set_xlim(-plotlim, plotlim)
    ax.set_ylim(-plotlim, plotlim)
    ax.set_zlim(-plotlim, plotlim)
    print(f"Saving figure at img/{fig_file}.fig.pickle")
    with open(f'img/{fig_file}.fig.pickle', 'wb') as file:
        pickle.dump(fig, file)
    if args['acc']:
        with open(args['sigma_file'], 'wb') as file:
            pickle.dump(sigmas, file)
    print(f"Elapsed time: {time.time() - tic:.2f} s")
    plt.show()


def main():
    args = load_yaml()
    plot_leaves(args)


if __name__ == '__main__':
    main()
