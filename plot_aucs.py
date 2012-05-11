#!/usr/bin/env python3

from collections import defaultdict
import itertools
import math
import pickle

import numpy as np

from plot_results import KEY_NAMES, ActivePMF # for pickle


def load_results(filenames):
    for filename in filenames:
        with open(filename, 'rb') as f:
            r = pickle.load(f)
        yield r

################################################################################

def rmse_auc(result):
    ns, rmses, ijs, evals = zip(*result)
    ns = np.asarray(ns)
    rmses = np.asarray(rmses)

    return ((rmses[:-1] + rmses[1:]) * np.diff(ns)).sum() / 2

def get_aucs(filenames):
    results = defaultdict(list)

    for r in load_results(filenames):
        for k, v in r.items():
            if not k.startswith('_'):
                results[k].append(rmse_auc(v))

    return {k: np.array(v) for k, v in results.items()}

def plot_aucs(filenames):
    import matplotlib.pyplot as plt

    names, aucs = zip(*sorted(
        (KEY_NAMES[k], v) for k, v in get_aucs(filenames).items()))
    aucs = np.array(aucs)

    if aucs.shape[1] == 1:
        plt.plot(aucs, linestyle='None', marker='o')
        indices = np.arange(len(names))
    else:
        plt.boxplot(aucs.T)
        indices = np.arange(len(names)) + 1
    plt.xticks(indices, names, rotation=90)
    plt.xlim(indices[0] - .5, indices[-1] + .5)
    plt.ylabel('AUC (RMSE)')
    plt.tight_layout()

def plot_cutoff_aucs(filenames, cutoff):
    import matplotlib.pyplot as plt

    names, aucs = zip(*sorted(
        (KEY_NAMES[k], v) for k, v in get_num_ge_cutoff_auc(filenames, cutoff).items()))
    aucs = np.array(aucs)

    if aucs.shape[1] == 1:
        plt.plot(aucs, linestyle='None', marker='o')
        indices = np.arange(len(names))
    else:
        plt.boxplot(aucs.T)
        indices = np.arange(len(names)) + 1
    plt.xticks(indices, names, rotation=90)
    plt.xlim(indices[0] - .5, indices[-1] + .5)
    plt.ylabel('AUC (# >= {})'.format(cutoff))
    plt.tight_layout()

################################################################################

def get_num_ge_cutoff(filenames, cutoff):
    results = defaultdict(list)

    des_ns = None

    for r in load_results(filenames):
        real = r['_real']

        for k, v in r.items():
            if k.startswith('_'):
                continue

            ns, rmses, ijs, evals = zip(*v)
            if des_ns is None:
                des_ns = ns
            else:
                assert np.all(ns == des_ns)

            assert ijs[0] is None
            poses = [(r['_ratings'][:,2] >= cutoff).sum()]

            for i, j in ijs[1:]:
                poses.append(poses[-1] + (1 if real[i,j] >= cutoff else 0))

            results[k].append(np.asarray(poses))

    return results, np.asarray(ns)


def get_num_ge_cutoff_mean(filenames, cutoff):
    results, ns = get_num_ge_cutoff(filenames, cutoff)
    return {k: np.mean(v, 0) for k, v in results.items()}, ns

def get_num_ge_cutoff_auc(filenames, cutoff):
    results, ns = get_num_ge_cutoff(filenames, cutoff)
    return {k: np.array([((poses[:-1] + poses[1:]) * np.diff(ns)).sum() / 2
                for poses in v])
            for k, v in results.items()}

def plot_num_ge_cutoff(filenames, cutoff):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    plt.xlabel("# of rated elements")
    plt.ylabel("# >= {}".format(cutoff))

    # cycle through colors and line styles
    colors = 'bgrcmyk'
    linestyles = ['-', '--', ':']
    l_c = itertools.cycle(itertools.product(linestyles, colors))

    # read in the actual results
    vals, ns = get_num_ge_cutoff(filenames, cutoff)
    ns = np.asarray(ns)
    nice_results = sorted((KEY_NAMES[k], v) for k, v in vals.items())

    # offset lines a bit so you can see when some of them overlap
    total = len(vals)
    offset = .15 / total

    for idx, (nice_name, vals) in enumerate(nice_results):
        nums = ns + (idx - total/2) * offset

        l, c = next(l_c)
        plt.plot(nums, vals, linestyle=l, color=c, label=nice_name, marker='^')

    # only show integer values for x ticks
    #xmin, xmax = plt.xlim()
    #plt.xticks(range(math.ceil(xmin), math.floor(xmax) + 1))

    plt.legend(loc='best', prop=FontProperties(size=9))


################################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')

    parser.add_argument('--auc', action='store_true', default=True)
    parser.add_argument('--no-auc', action='store_false', dest='auc')

    parser.add_argument('--ge-cutoff', nargs='+', type=float)
    parser.add_argument('--ge-cutoff-auc', nargs='+', type=float)

    #parser.add_argument('--save')
    args = parser.parse_args()

    #if args.save:
    #    import matplotlib
    #    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    if args.auc:
        plt.figure()
        plot_aucs(args.files)

    if args.ge_cutoff:
        for cutoff in args.ge_cutoff:
            plt.figure()
            plot_num_ge_cutoff(args.files, cutoff)

    if args.ge_cutoff_auc:
        for cutoff in args.ge_cutoff_auc:
            plt.figure()
            plot_cutoff_aucs(args.files, cutoff)

    plt.show()
