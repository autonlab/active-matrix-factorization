#!/usr/bin/env python3

from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from plot_results import KEY_NAMES, ActivePMF # for pickle


def rmse_auc(result):
    ns, rmses, ijs, evals = zip(*result)
    ns = np.asarray(ns)
    rmses = np.asarray(rmses)

    return ((rmses[:-1] + rmses[1:]) * np.diff(ns)).sum() / 2

def get_aucs(filenames):
    results = defaultdict(list)

    for filename in filenames:
        with open(filename, 'rb') as f:
            r = pickle.load(f)

        for k, v in r.items():
            if not k.startswith('_'):
                results[k].append(rmse_auc(v))

    return {k: np.array(v) for k, v in results.items()}


def plot_aucs(filenames):
    import matplotlib.pyplot as plt

    names, aucs = zip(*sorted(
        (KEY_NAMES[k], v) for k, v in get_aucs(filenames).items()))

    plt.boxplot(aucs)
    plt.xticks(np.arange(len(names)) + 1, names, rotation=90)
    plt.tight_layout()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plot_aucs(sys.argv[1:])
    plt.show()
