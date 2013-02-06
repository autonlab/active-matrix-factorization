#!/usr/bin/env python3

import itertools
import os

import numpy as np
from scipy.stats import kendalltau
from statsmodels.graphics.correlation import plot_corr

from plot_results import (load_results, plot_criteria_firsts, add_bool_opt,
                          ActivePMF, BayesianPMF, BPMF)  # for pickle


def main():
    import matplotlib.pyplot as plt

    import argparse
    parser = argparse.ArgumentParser()

    add_bool_opt(parser, 'share-cbar', default=True)

    parser.add_argument('--types', nargs='*',
                        default=('bayes', 'stan', 'stan_straightforward'))

    parser.add_argument('dirname')
    parser.add_argument('key')

    args = parser.parse_args()

    results = {
        t: load_results(os.path.join(args.dirname, 'results_{}.pkl'.format(t)))
        for t in args.types
    }
    kn = lambda t: '{}_{}'.format(t, args.key)
    items = [(kn(t), results[t][kn(t)]) for t in args.types]

    plot_criteria_firsts(items, share_cbar=args.share_cbar)

    firsts = np.asarray([res[1][-1] for name, res in items])
    firsts = firsts.reshape(firsts.shape[0], -1)

    # remove nan elements to avoid messing with correlation measures
    nans = np.isnan(firsts)
    assert np.all(np.all(nans, axis=0) == np.any(nans, axis=0))
    firsts = firsts[:, np.logical_not(nans[0])]

    n = firsts.shape[0]
    rho = np.zeros((n, n))
    p = np.zeros((n, n))
    for a, b in itertools.combinations_with_replacement(range(n), 2):
        rho_, p_ = kendalltau(firsts[a], firsts[b])
        rho[a, b] = rho[b, a] = rho_
        p[a, b] = p[b, a] = p_
    plot_corr(rho, normcolor=True, xnames=args.types, title='Kendall Tau')

    plt.show()


if __name__ == '__main__':
    main()
