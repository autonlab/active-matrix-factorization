#!/usr/bin/env python3

# TODO: figure out origin / transpose / etc bullshit

import itertools
import math
from operator import itemgetter
import os
import re
import sys
import warnings

import numpy as np

_dirname = os.path.dirname(__file__)

# imports to make pickle happy
sys.path.append(os.path.join(_dirname, 'python-pmf'))
import active_pmf
from active_pmf import ActivePMF
import mn_active_pmf
from mn_active_pmf import MNActivePMF
import bayes_pmf
from bayes_pmf import BayesianPMF

sys.path.append(os.path.join(_dirname, 'stan-bpmf'))
import bpmf
from bpmf import BPMF
from bpmf_newitems import NewItemsBPMF

sys.path.append(os.path.join(_dirname, 'ratingconcentration'))
import active_rc

sys.path.append(os.path.join(_dirname, 'mmmf'))
import active_mmmf

KEY_NAMES = {k: f.nice_name for k, f in active_pmf.KEY_FUNCS.items()}
KEY_NAMES.update({'mnpmf_' + k: 'MN: ' + f.nice_name
                  for k, f in mn_active_pmf.KEY_FUNCS.items()})
KEY_NAMES.update({'rc_' + k: 'RC: ' + f.nice_name
                  for k, f in active_rc.KEY_FUNCS.items()})
KEY_NAMES.update({'mmmf_' + k: 'MMMF: ' + f.nice_name
                  for k, f in active_mmmf.KEY_FUNCS.items()})
KEY_NAMES.update({'bayes_' + k: 'Bayes: ' + f.nice_name
                  for k, f in bayes_pmf.KEYS.items()})
KEY_NAMES.update({'stan_' + k: 'Stan: ' + f.nice_name
                  for k, f in bpmf.KEYS.items()})
KEY_NAMES.update({'stan_straightforward_' + k: 'SStan: ' + f.nice_name
                  for k, f in bpmf.KEYS.items()})
KINDS = {'apmf', 'mnpmf', 'rc', 'mmmf', 'bayes', 'stan', 'stan_straightforward'}


from matplotlib import cm
default_cmap = cm.cool


def auc_roc(dec, label):
    # number of negatives, positives
    posneg = np.array([np.sum(label == 1), np.sum(label == 0)])
    assert posneg.sum() == label.size
    assert np.all(np.isfinite(dec))

    dl = np.zeros(label.size, dtype=[('dec', float), ('label', bool)])
    dl['dec'] = dec
    dl['label'] = label
    dl.sort(order='dec')

    if np.any(posneg == 0):
        return 0, None
        # TODO FIXME

    tpfp = np.zeros(2)  # true, false positives so far in the sweep
    roc_pts = [np.zeros(2)]  # points on the ROC curve

    # sweep over possible thresholds
    for d, group in itertools.groupby(dl, key=itemgetter(0)):
        # group is an iterator over labels with this threshold
        for _, l in group:
            tpfp[1 - int(l)] += 1

        roc_pts.append(tpfp / posneg)

    roc_pts = np.asarray(roc_pts)

    xs, ys = roc_pts.T
    return np.trapz(x=xs, y=ys), roc_pts


################################################################################
### Plotting code

def plot_predictions(apmf, real, cmap=default_cmap):
    from matplotlib import pyplot as plt

    pred = apmf.predicted_matrix()
    a_mean, a_var = apmf.approx_pred_means_vars()
    a_std = np.sqrt(a_var)

    xs = (real, pred, a_mean)
    norm = plt.Normalize(min(a.min() for a in xs), max(a.max() for a in xs))

    rated = np.array(list(apmf.rated))

    def show(mat, title, subplot, norm_=norm):
        plt.subplot(subplot)

        plt.imshow(mat, norm=norm_, cmap=cmap, interpolation='nearest',
                origin='lower')
        plt.colorbar()
        plt.title(title)

        if apmf.rated:
            plt.scatter(rated[:, 1], rated[:, 0], marker='s', s=15, c='white')

    show(real, "Real", 221)
    show(pred, "MAP", 222)
    show(a_mean, "Normal: Mean", 223)
    show(a_std, "Normal: Std Dev", 224, plt.Normalize(0, a_std.max()))


def plot_real(real, rated=None, cmap=default_cmap):
    from matplotlib import pyplot as plt

    plt.imshow(real, cmap=cmap, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title("True Matrix")
    if rated is not None:
        plt.scatter(rated[:, 1], rated[:, 0], marker='s', s=15, c='white')


def linestyle_color_marker(num=0):
    from itertools import cycle

    linestyles = ('-', '--')
    colors = 'bgrck'
    if num < 100:
        markers = ('o', '^', 's')
    else:
        markers = [None]
    return zip(cycle(linestyles), cycle(colors), cycle(markers))


def _plot_lines(results, fn, ylabel):
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    plt.xlabel("# of rated elements")
    plt.ylabel(ylabel)

    # cycle through colors and line styles
    l_c_m = linestyle_color_marker()

    # offset lines a bit so you can see when some of them overlap
    total = len(results)
    offset = .15 / total

    nice_results = ((KEY_NAMES[k], k, v)
                    for k, v in results.items() if not k.startswith('_'))

    for idx, (nice_name, key_name, result) in enumerate(sorted(nice_results)):
        if len(result[0]) == 4:
            nums, rmses, ijs, vals = zip(*result)
            preds = [None for _ in nums]
        else:
            nums, rmses, ijs, vals, preds = zip(*result)
        vals = fn(nums, rmses, ijs, vals, preds, results)
        nums = np.array(nums, copy=False) + (idx - total / 2) * offset

        l, c, m = next(l_c_m)
        plt.plot(nums, vals, linestyle=l, color=c, label=nice_name, marker=m)

    # only show integer values for x ticks
    xmin, xmax = plt.xlim()
    if xmax - xmin < 20:
        plt.xticks(range(math.ceil(xmin), math.floor(xmax) + 1))

    plt.legend(loc='best', prop=FontProperties(size=9))


def plot_rmses(results, keys):
    def get_rmses(nums, rmses, ijs, vals, preds, results):
        return rmses
    _plot_lines({k: v for k, v in results.items() if k in keys},
                get_rmses, "RMSE")

def plot_pred_aucs(results, keys):
    def get_aucs(nums, rmses, ijs, vals, preds, results):
        test_on = results['_test_on']
        label = results['_real'][test_on] > 0
        aucs = np.array([
            np.nan if pred is None else auc_roc(pred[test_on], label)[0]
            for pred in preds
        ])
        return aucs
    _plot_lines({k: v for k, v in results.items()
                      if k in keys or k.startswith('_')},
                get_aucs, "Classification AUCs")


def plot_num_ge_cutoff(results, cutoff, keys):
    def get_cutoffs(nums, rmses, ijs, vals, preds, results):
        real = results['_real']

        assert ijs[0] is None
        ns = [(results['_ratings'][:, 2] >= cutoff).sum()]

        for i, j in ijs[1:]:
            ns.append(ns[-1] + (1 if real[i, j] >= cutoff else 0))

        return ns

    _plot_lines({k: v for k, v in results.items() if k in keys},
                get_cutoffs, "# found > {}".format(cutoff))


def subplot_config(n):
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)
    return nr, nc


def plot_criteria_over_time(name, result, cmap=default_cmap):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    if len(result[0]) == 4:
        nums, rmses, ijs, valses = zip(*result)
    else:
        nums, rmses, ijs, valses, preds = zip(*result)

    assert ijs[0] is None
    assert valses[0] is None
    ijs = ijs[1:]
    valses = valses[1:]

    if valses[-1] is None or np.all(np.isnan(valses[-1])):
        ijs = ijs[:-1]
        valses = valses[:-1]

    nr, nc = subplot_config(len(ijs))

    fig = plt.figure()
    #fig.suptitle(name)
    grid = ImageGrid(fig, 111, nrows_ncols=(nr, nc), axes_pad=.3,
            cbar_location='right', cbar_mode='single')

    n_users, n_items = valses[0].shape
    xticks = np.linspace(-.5, n_items - .5, n_items + 1)
    yticks = np.linspace(-.5, n_users - .5, n_users + 1)

    finite_vals = [vals[np.isfinite(vals)] for vals in valses]
    vmin = min(f_vals.min() for f_vals in finite_vals if f_vals.size)
    vmax = max(f_vals.max() for f_vals in finite_vals if f_vals.size)
    norm = plt.Normalize(vmin, vmax)
    # TODO: dynamically adjust color range to be more distinguishable?

    for idx, (n, rmse, (i,j), vals) in enumerate(zip(nums, rmses, ijs, valses)):
        # we know n values and have RMSE of rmse, then pick ij based on vals

        #grid[idx].set_title("{}".format(n + 1))

        im = grid[idx].imshow(vals, interpolation='nearest', cmap=cmap,
                   origin='upper', aspect='equal', norm=norm)

        grid[idx].set_xticks(xticks)
        grid[idx].set_yticks(yticks)
        grid[idx].set_xticklabels([])
        grid[idx].set_yticklabels([])
        grid[idx].set_xlim(xticks[0], xticks[-1])
        grid[idx].set_ylim(yticks[0], yticks[-1])
        grid[idx].grid()

        # mark the selected point (indices are transposed)
        grid[idx].scatter(j, i, marker='s', c='white', s=50)  # s=15)

    for idx in range(len(ijs), nr * nc):
        grid[idx].set_visible(False)

    grid.cbar_axes[0].colorbar(im)

    return fig


def plot_criteria_firsts(result_items, cmap=default_cmap, share_cbar=False):
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1 import ImageGrid

    prop = FontProperties(size=9)
    nr, nc = subplot_config(len(result_items))

    fig = plt.figure()
    fig.suptitle("Criteria First Steps")

    if share_cbar:
        vmin = np.inf
        vmax = -np.inf
        for name, data in result_items:
            vals = data[1][-1]
            vmin = min(vmin, np.nanmin(vals))
            vmax = max(vmax, np.nanmax(vals))
        cbar_args = dict(cbar_location='right', cbar_mode='single', cbar_pad=.5)
    else:
        vmin = vmax = None
        cbar_args = dict(cbar_location='right', cbar_mode='each', cbar_pad=.1)

    grid = ImageGrid(fig, 111, nrows_ncols=(nr, nc), axes_pad=.5, **cbar_args)

    n_users, n_items = result_items[0][1][1][3].shape
    xticks = np.linspace(-.5, n_items - .5, n_items + 1)
    yticks = np.linspace(-.5, n_users - .5, n_users + 1)

    for idx, (name, data) in enumerate(result_items):
        assert data[0][3] is None

        n, rmse, (i, j), vals, *rest = data[1]

        im = grid[idx].matshow(vals, cmap=cmap, origin='upper', aspect='equal',
                               vmin=vmin, vmax=vmax)

        grid[idx].set_title(KEY_NAMES.get(name, name), font_properties=prop)
        grid[idx].set_xticks(xticks)
        grid[idx].set_yticks(yticks)
        grid[idx].set_xticklabels([])
        grid[idx].set_yticklabels([])
        grid[idx].set_xlim(xticks[0], xticks[-1])
        grid[idx].set_ylim(yticks[0], yticks[-1])
        grid[idx].grid()

        grid[idx].scatter(j, i, marker='s', c='white', s=20)
        grid[idx].cax.colorbar(im)

    for idx in range(len(result_items), nr * nc):
        grid[idx].set_visible(False)
        grid.cbar_axes[idx].set_visible(False)

    return fig


################################################################################
### Command-line interface

def add_bool_opt(parser, name, default=False):
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--' + name, action='store_true', default=default)
    g.add_argument('--no-' + name, action='store_false',
            dest=name.replace('-', '_'))

def guess_kind(filename):
    kinds = re.compile(r'results_({})'.format(
        '|'.join(re.escape(k) for k in sorted(KINDS, key=len, reverse=True))))
    match = kinds.search(filename)
    return match.group(1) if match else 'apmf'

_warned_about = set()
def load_results(filename, kind=None):
    r = np.load(filename)

    if kind is None:
        kind = guess_kind(filename)

    if all(k.startswith('_') for k in r):
        if filename not in _warned_about:
            print("WARNING: No data in {}".format(filename), file=sys.stderr)
        _warned_about.add(filename)

    if kind == 'apmf':
        return {k: v for k, v in r.items()}
    else:
        rep = re.compile(r'^(?!(_|{}_))'.format(kind))
        return {rep.sub(kind + '_', k): v for k, v in r.items()}


def main(argstr=None):
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()

    parser.add_argument('results_file', metavar='FILE')
    parser.add_argument('keys', nargs='*',
            help="Choices: {}.".format(', '.join(sorted(KEY_NAMES))))

    add_bool_opt(parser, 'real', False)
    add_bool_opt(parser, 'rmse', False)
    add_bool_opt(parser, 'pred-auc', False)
    parser.add_argument('--cutoff', type=float, nargs='+', metavar='CUTOFF')
    add_bool_opt(parser, 'criteria', False)
    add_bool_opt(parser, 'criteria-firsts', False)
    add_bool_opt(parser, 'initial-preds', False)

    parser.add_argument('--kind', default=None, choices=KINDS)

    parser.add_argument('--all-plots', default=False, action='store_true')

    parser.add_argument('--cmap', default=default_cmap, type=cm.get_cmap)
    parser.add_argument('--filetype', default='png')
    parser.add_argument('--outdir', nargs='?', const=True, default=None,
            metavar='DIR')
    add_bool_opt(parser, 'interactive', None)

    if argstr is not None:
        import shlex
        args = parser.parse_args(shlex.split(argstr))
    else:
        args = parser.parse_args()

    if args.all_plots:
        args.real = True
        args.rmse = True
        args.criteria = True
        args.criteria_firsts = True
        args.initial_preds = True

    # try to make the out directory if necessary; set up save_plot fn
    if args.outdir:
        if args.outdir is True:
            args.outdir = os.path.dirname(args.results_file)

        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        def save_plot(name, fig=None):
            if fig is None:
                from matplotlib import pyplot as fig
            fname = os.path.join(args.outdir, name)
            fig.savefig(fname + '.' + args.filetype,
                        bbox_inches='tight', pad_inches=.1)
    else:
        def save_plot(name, fig=None):
            pass

    # interactive default depends on if outdir is present
    if args.interactive is None:
        args.interactive = not args.outdir

    # load the results
    results = load_results(args.results_file, kind=args.kind)

    # check args.keys are actually in the results
    if not args.keys:
        args.keys = list(k for k in results.keys() if not k.startswith('_'))
    else:
        good_keys = []
        for k in args.keys:
            if k in results:
                good_keys.append(k)
            else:
                warnings.warn("WARNING: requested key {} not in the saved "
                              "results.".format(k))
        args.keys = good_keys

    # use agg backend if we're not showing these interactively
    if not args.interactive:
        import matplotlib
        matplotlib.use('Agg')

    from matplotlib import pyplot as plt

    # real data plot
    if args.real:
        print("Plotting real matrix")
        fig = plt.figure()
        plot_real(results['_real'], results['_ratings'], cmap=args.cmap)
        save_plot('real', fig)

    # RMSE plot
    if args.rmse:
        print("Plotting RMSEs")
        fig = plt.figure()
        plot_rmses(results, args.keys)
        save_plot('rmses', fig)

    # binary classification AUCs
    if args.pred_auc:
        print("Plotting binary classification AUCs")
        fig = plt.figure()
        plot_pred_aucs(results, args.keys)
        save_plot('pred_aucs', fig)

    # plot of numbers >= a cutoff
    if args.cutoff is not None:
        for cutoff in args.cutoff:
            print("Plotting cutoff {}".format(cutoff))
            fig = plt.figure()
            plot_num_ge_cutoff(results, cutoff, args.keys)
            save_plot('ge-{}'.format(cutoff), fig)

    # plot of each criterion
    if args.criteria:
        for criterion in args.keys:
            result = results[criterion]
            nice_name = KEY_NAMES[criterion]
            print("Plotting {}".format(nice_name))

            fig = plot_criteria_over_time(nice_name, result, cmap=args.cmap)
            save_plot('{}'.format(criterion), fig)

    # plot of criteria first steps
    if args.criteria_firsts:
        print("Plotting criteria first steps")
        items = sorted(
                ((k, v) for k, v in results.items() if k in args.keys),
                key=lambda item: KEY_NAMES[item[0]])

        fig = plot_criteria_firsts(items, cmap=args.cmap)
        save_plot('firsts', fig)

    # plot of initial predictions
    if args.initial_preds:
        apmf = results.get('_initial_apmf', None)
        if not apmf:
            print("Can't do initial predictions: not in the file")
        else:
            print("Plotting initial predictions")
            fig = plt.figure()
            plot_predictions(apmf, results['_real'], cmap=args.cmap)
            save_plot('initial_preds', fig)

    # pause to look at plots if interactive
    if args.interactive:
        plt.show()

if __name__ == '__main__':
    main()
