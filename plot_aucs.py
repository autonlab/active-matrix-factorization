#!/usr/bin/env python3

from collections import defaultdict
import functools
import itertools
import math
import re
import sys

import numpy as np

from plot_results import (KEY_NAMES, ActivePMF, BayesianPMF, BPMF,  # for pickle
                          linestyle_color_marker, load_results)

################################################################################
### loading data

def load_data(filenames, do_rmse=False, do_rmse_auc=False,
                         do_cutoffs=None, do_cutoff_aucs=None):
    desired_ns = None

    want_rmses = do_rmse or do_rmse_auc
    if want_rmses:
        rmse_traces = defaultdict(list)

    cutoff_vals = set()
    if do_cutoffs:
        cutoff_vals.update(do_cutoffs)
    if do_cutoff_aucs:
        cutoff_vals.update(do_cutoff_aucs)

    if cutoff_vals:
        cutoff_traces = defaultdict(functools.partial(defaultdict, list))

    if not cutoff_vals and not results:  # not asked to do anything!
        return {}

    for r in map(load_results, filenames):
        if cutoff_vals:
            real = r['_real']
            ratings = r['_ratings']

        for k, v in r.items():
            if k.startswith('_'):
                continue

            ns, rmses, ijs, evals = zip(*v)
            ns = np.asarray(ns)
            if desired_ns is not None:
                assert np.all(ns == desired_ns)
            else:
                desired_ns = ns

            if want_rmses:
                rmse_traces[k].append(rmses)

            if cutoff_vals:
                assert ijs[0] is None
                for cutoff in cutoff_vals:
                    poses = [(ratings[:, 2] >= cutoff).sum()]
                    for i, j in ijs[1:]:
                        poses.append(1 if real[i, j] >= cutoff else 0)
                    cutoff_traces[cutoff][k].append(np.cumsum(poses))

    results = {'ns': desired_ns}

    if do_rmse:  # name => mean RMSE curve
        results['rmse'] = {
            k: np.mean(v, axis=0) for k, v in rmse_traces.items()
        }

    if do_rmse_auc:  # name => array of area under RMSE curves
        results['rmse_auc'] = {
            k: np.trapz(v, axis=1) for k, v in rmse_traces.items()
        }

    if do_cutoffs:  # cutoff => name => curve of # positives
        results['cutoffs'] = {
            cutoff: {k: np.mean(v, axis=0) for k, v in c_vals.items()}
            for cutoff, c_vals in cutoff_traces.items()
        }

    if do_cutoff_aucs:  # cutoff => name => array of AUC for curves of # pos
        results['cutoff_aucs'] = {
            cutoff: {k: np.trapz(v, axis=1) for k, v in c_vals.items()}
            for cutoff, c_vals in cutoff_traces.items()
        }

    return results


################################################################################
### plotting

def show_legend(where='outside', fontsize=11):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    if where == 'outside':
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * .7, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, .5),
                   prop=FontProperties(size=10))
    else:
        plt.legend(loc='best', prop=FontProperties(size=fontsize))


def plot_lines(ns, data, ylabel=None):
    import matplotlib.pyplot as plt

    plt.xlabel("# of rated elements")
    plt.ylabel(ylabel)

    nice_results = sorted(
            ((KEY_NAMES[k], v) for k, v in data.items()),
            key=lambda kv: kv[1][-1], reverse=True)

    # offset lines a bit so you can see when some of them overlap
    total = len(ns)
    offset = .15 / total if total > 3 else .02

    # cycle through colors and line styles
    l_c_m = linestyle_color_marker(len(ns))

    for idx, (nice_name, vals) in enumerate(nice_results):
        nums = ns + (idx - total / 2) * offset

        l, c, m = next(l_c_m)
        plt.plot(nums, vals, linestyle=l, color=c, label=nice_name, marker=m)

    # only show integer values for x ticks
    #xmin, xmax = plt.xlim()
    #plt.xticks(range(math.ceil(xmin), math.floor(xmax) + 1))


def plot_aucs(aucs, ylabel=None):
    import matplotlib.pyplot as plt
    names, aucs = list(zip(*sorted((KEY_NAMES[k], v) for k, v in aucs.items())))

    if all(a.size == 1 for a in aucs):
        plt.plot(aucs, linestyle='None', marker='o')
        indices = np.arange(len(names))
    else:
        try:
            from statsmodels.graphics.boxplots import beanplot
        except ImportError:
            plt.boxplot(aucs)
        else:
            beanplot(aucs, ax=plt.gca(), plot_opts={'cutoff': True})
        indices = np.arange(len(names)) + 1
    plt.xticks(indices, names, rotation=90)
    plt.xlim(indices[0] - .5, indices[-1] + .5)
    plt.ylabel(ylabel)
    plt.tight_layout()

################################################################################


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--rmses', action='store_true', default=False)
    g.add_argument('--no-rmses', action='store_false', dest='rmses')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--auc', action='store_true', default=True)
    g.add_argument('--no-auc', action='store_false', dest='auc')

    parser.add_argument('--ge-cutoff', nargs='+', type=float)
    parser.add_argument('--ge-cutoff-auc', nargs='+', type=float)

    parser.add_argument('--legend', default='outside',
                        choices={'outside', 'inside'})

    #parser.add_argument('--save')
    args = parser.parse_args()

    #if args.save:
    #    import matplotlib
    #    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    data = load_data(args.files,
        do_rmse=args.rmses, do_rmse_auc=args.auc,
        do_cutoffs=args.ge_cutoff, do_cutoff_aucs=args.ge_cutoff_auc)
    ns = data['ns']

    if args.rmses:
        plt.figure()
        plot_lines(ns, data['rmse'], 'RMSE')
        show_legend(args.legend)

    if args.auc:
        plt.figure()
        plot_aucs(data['rmse_auc'], 'AUC (RMSE)')

    if args.ge_cutoff:
        for cutoff in args.ge_cutoff:
            plt.figure()
            plot_lines(ns, data['cutoffs'][cutoff], '# >= {}'.format(cutoff))
            show_legend(args.legend)

    if args.ge_cutoff_auc:
        for cutoff in args.ge_cutoff_auc:
            plt.figure()
            plot_aucs(data['cutoff_aucs'][cutoff],
                      'AUC (# >= {})'.format(cutoff))

    plt.show()

if __name__ == '__main__':
    main()
