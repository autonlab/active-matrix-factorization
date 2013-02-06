#!/usr/bin/env python3

import functools
import itertools
import os

import numpy as np
from scipy import stats
from scipy.spatial import distance

from plot_results import (load_results, plot_criteria_firsts, add_bool_opt,
        guess_kind, default_cmap, ActivePMF, BayesianPMF, BPMF)  # for pickle

from matplotlib import cm


def get_pairwise(fn, vals):
    n = vals.shape[0]
    res = np.zeros((n, n))
    for a, b in itertools.combinations_with_replacement(range(n), 2):
        res[a, b] = res[b, a] = fn(vals[a], vals[b])
    return res


def imshow_with_names(vals, names, vmin=None, vmax=None, cmap=None,
                      ax=None, title=None):
    # based on statsmodels.graphics.correlation.plot_corr
    if ax is None:
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = ax.figure

    n = len(names)
    assert vals.shape == (n, n)
    im = ax.imshow(vals, vmin=vmin, vmax=vmax, cmap=cmap,
                   extent=(0, n, 0, n), interpolation='nearest',
                   aspect='auto')

    l_pos = np.arange(0, n) + 0.5
    ax.set_xticks(l_pos)
    ax.set_yticks(l_pos)

    ax.set_xticks(l_pos[:-1] + 0.5, minor=True)
    ax.set_yticks(l_pos[:-1] + 0.5, minor=True)

    ax.set_xticklabels(names, rotation=45, ha='right', fontsize='small')
    ax.set_yticklabels(names[::-1], ha='right', fontsize='small')

    if title:
        ax.set_title(title)

    fig.colorbar(im, use_gridspec=True)
    fig.tight_layout()

    ax.tick_params(which='minor', length=0)
    ax.tick_params(direction='out', top=False, right=False)
    ax.grid(True, which='minor', linestyle='-', color='w', lw=1)

    return ax


def beanplot_grid(vals, names, fig=None, title=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from statsmodels.graphics.boxplots import beanplot

    if fig is None:
        fig = plt.figure()

    if title:
        fig.suptitle(title)

    n = vals.shape[1]
    grid = ImageGrid(fig, 111, nrows_ncols=(n, n), aspect=False,
                     share_all=True, label_mode='L')
    for i in range(n * n):
        grid[i].set_visible(False)

    for j, i in itertools.combinations_with_replacement(range(n), 2):
        idx = i * n + j
        grid[idx].set_visible(True)
        data = vals[:, i, j]
        if np.ptp(data) == 0:
            grid[idx].hlines(data[0], .85, 1.15, lw=0.5, color='k')
        else:
            beanplot([data], labels=[''], ax=grid[idx],
                     plot_opts={'cutoff': True})

    # TODO: actually show the names...
    return grid


def load_from_dir(dirname, names, key):
    filenames = {n: os.path.join(dirname, 'results_{}.pkl'.format(n))
                 for n in names}
    results = {n: load_results(filenames[n]) for n in names}

    kinds = {n: guess_kind(filenames[n]) for n in names}
    kn = lambda n: '{}_{}'.format(kinds[n], key)

    return [(n, results[n][kn(n)]) for n in names]


def grab_nonnan_firsts(items):
    firsts = np.asarray([res[1][-1] for name, res in items])
    firsts = firsts.reshape(firsts.shape[0], -1)

    nans = np.isnan(firsts)
    assert np.all(np.all(nans, axis=0) == np.any(nans, axis=0))
    return firsts[:, np.logical_not(nans[0])]


def main():
    import matplotlib.pyplot as plt

    import argparse
    parser = argparse.ArgumentParser()

    add_bool_opt(parser, 'share-cbar', default=True)

    parser.add_argument('--names', nargs='*',
        default=('bayes', 'stan', 'stan_straightforward'))

    parser.add_argument('--cmap', default=default_cmap, type=cm.get_cmap)

    parser.add_argument('key')
    parser.add_argument('dirnames', nargs='+')

    args = parser.parse_args()

    get_kendall = functools.partial(get_pairwise,
        lambda *a, **kw: stats.kendalltau(*a, **kw)[0])
    get_rms = lambda evals: distance.squareform(
        distance.pdist(evals, 'euclidean') / np.sqrt(evals.shape[0]))

    if len(args.dirnames) == 1:
        items = load_from_dir(args.dirnames[0], args.names, args.key)
        plot_criteria_firsts(items, share_cbar=args.share_cbar, cmap=args.cmap)

        firsts = grab_nonnan_firsts(items)

        # kendall's tau (measure of rank correlation)
        tau = get_kendall(firsts)
        imshow_with_names(tau, args.names, title="Kendall's Tau",
                          vmin=-1, vmax=1, cmap='RdYlBu_r')

        # RMS distance between evaluations
        rms = get_rms(firsts)
        imshow_with_names(rms, args.names, title="RMS distance",
                          cmap='hot', vmin=0, vmax=1.2 * rms.max())

    else:
        firsts_by_dir = [
            grab_nonnan_firsts(load_from_dir(d, args.names, args.key))
            for d in args.dirnames
        ]

        taus = np.array([get_kendall(evals) for evals in firsts_by_dir])
        beanplot_grid(taus, args.names, title="Kendall's Tau")

        rmses = np.array([get_rms(evals) for evals in firsts_by_dir])
        beanplot_grid(rmses, args.names, title="RMS distances")

    plt.show()


if __name__ == '__main__':
    main()
