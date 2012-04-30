#!/usr/bin/env python3

# TODO: figure out origin / transpose / etc bullshit

import itertools
import math
import os
import sys
import warnings

import numpy as np

_dirname = os.path.dirname(__file__)

sys.path.append(os.path.join(_dirname, 'python-pmf'))
import active_pmf
from active_pmf import ActivePMF # to make pickle happy

sys.path.append(os.path.join(_dirname, 'ratingconcentration'))
import active_rc

KEY_NAMES = { k: f.nice_name for k, f in active_pmf.KEY_FUNCS.items() }
KEY_NAMES.update({'rc_'+k: 'RC: '+f.nice_name
                  for k, f in active_rc.KEY_FUNCS.items()})


################################################################################
### Plotting code

def plot_predictions(apmf, real):
    from matplotlib import pyplot as plt
    from matplotlib import cm

    pred = apmf.predicted_matrix()
    a_mean, a_var = apmf.approx_pred_means_vars()
    a_std = np.sqrt(a_var)

    xs = (real, pred, a_mean)
    norm = plt.Normalize(min(a.min() for a in xs), max(a.max() for a in xs))

    rated = np.array(list(apmf.rated))
    def show(mat, title, subplot, norm_=norm):
        plt.subplot(subplot)

        plt.imshow(mat, norm=norm_, cmap=cm.jet, interpolation='nearest',
                origin='lower')
        plt.colorbar()
        plt.title(title)

        if apmf.rated:
            plt.scatter(rated[:,1], rated[:,0], marker='s', s=15, c='white')

    show(real, "Real", 221)
    show(pred, "MAP", 222)
    show(a_mean, "Normal: Mean", 223)
    show(a_std, "Normal: Std Dev", 224, plt.Normalize(0, a_std.max()))


def _plot_lines(results, fn, ylabel):
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    plt.xlabel("# of rated elements")
    plt.ylabel(ylabel)

    # cycle through colors and line styles
    colors = 'bgrcmyk'
    linestyles = ['-', '--', ':']
    l_c = itertools.cycle(itertools.product(linestyles, colors))

    # offset lines a bit so you can see when some of them overlap
    total = len(results)
    offset = .15 / total

    nice_results = ((KEY_NAMES[k], k, v)
                    for k, v in results.items() if not k.startswith('_'))

    for idx, (nice_name, key_name, result) in enumerate(sorted(nice_results)):
        nums, rmses, ijs, vals = zip(*result)
        vals = fn(nums, rmses, ijs, vals, results)
        nums = np.array(nums, copy=False) + (idx - total/2) * offset

        l, c = next(l_c)
        plt.plot(nums, vals, linestyle=l, color=c, label=nice_name, marker='^')

    # only show integer values for x ticks
    xmin, xmax = plt.xlim()
    plt.xticks(range(math.ceil(xmin), math.floor(xmax) + 1))

    plt.legend(loc='best', prop=FontProperties(size=9))

def plot_rmses(results):
    def get_rmses(nums, rmses, ijs, vals, results):
        return rmses
    _plot_lines(results, get_rmses, "RMSE")

def plot_num_ge_cutoff(results, cutoff):
    def get_cutoffs(nums, rmses, ijs, vals, results):
        real = results['_real']

        assert ijs[0] is None
        ns = [(results['_ratings'][:,2] >= cutoff).sum()]

        for i, j in ijs[1:]:
            ns.append(ns[-1] + (1 if real[i,j] >= cutoff else 0))

        return ns

    _plot_lines(results, get_cutoffs, "# found > {}".format(cutoff))


def subplot_config(n):
    if n <= 3:
        return 1, n
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)
    return nr, nc


def plot_criteria_over_time(name, result, cmap=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    if cmap is None:
        from matplotlib import cm
        cmap = cm.jet

    nums, rmses, ijs, valses = zip(*result)

    assert ijs[0] is None
    assert valses[0] is None
    ijs = ijs[1:]
    valses = valses[1:]

    if valses[-1] is None or np.all(np.isnan(valses[-1])):
        ijs = ijs[:-1]
        valses = valses[:-1]

    nr, nc = subplot_config(len(ijs))

    fig = plt.figure()
    fig.suptitle(name)
    grid = ImageGrid(fig, 111, nrows_ncols=(nr,nc), axes_pad=.3,
            cbar_location='right', cbar_mode='single')

    n_users, n_items = valses[0].shape
    xticks = np.linspace(-.5, n_items - .5, n_items + 1)
    yticks = np.linspace(-.5, n_users - .5, n_users + 1)

    vmin = min(vals[np.isfinite(vals)].min() for vals in valses)
    vmax = max(vals[np.isfinite(vals)].max() for vals in valses)
    norm = plt.Normalize(vmin, vmax)
    # TODO: dynamically adjust color range to be more distinguishable?

    for idx, (n, rmse, (i,j), vals) in enumerate(zip(nums, rmses, ijs, valses)):
        # we know n values and have RMSE of rmse, then pick ij based on vals

        grid[idx].set_title("{}: ({:.3})".format(n + 1, rmse))

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
        grid[idx].scatter(j, i, marker='s', c='white', s=15)

    for idx in range(len(ijs), nr * nc):
        grid[idx].set_visible(False)

    grid.cbar_axes[0].colorbar(im)

    return fig

def plot_criteria_firsts(result_items, cmap=None):
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1 import ImageGrid
    if cmap is None:
        from matplotlib import cm
        cmap = cm.jet

    prop = FontProperties(size=9)
    nr, nc = subplot_config(len(result_items))

    fig = plt.figure()
    fig.suptitle("Criteria First Steps")
    grid = ImageGrid(fig, 111, nrows_ncols=(nr,nc), axes_pad=.5,
            cbar_pad=.1, cbar_location='right', cbar_mode='each')

    n_users, n_items = result_items[0][1][1][3].shape
    xticks = np.linspace(-.5, n_items - .5, n_items + 1)
    yticks = np.linspace(-.5, n_users - .5, n_users + 1)

    for idx, (name, data) in enumerate(result_items):
        assert data[0][3] is None

        n, rmse, (i,j), vals = data[1]

        im = grid[idx].imshow(vals, interpolation='nearest', cmap=cmap,
                origin='upper', aspect='equal')

        grid[idx].set_title(KEY_NAMES[name], font_properties=prop)
        grid[idx].set_xticks(xticks)
        grid[idx].set_yticks(yticks)
        grid[idx].set_xticklabels([])
        grid[idx].set_yticklabels([])
        grid[idx].set_xlim(xticks[0], xticks[-1])
        grid[idx].set_ylim(yticks[0], yticks[-1])
        grid[idx].grid()

        grid[idx].scatter(j, i, marker='s', c='white', s=20)
        grid.cbar_axes[idx].colorbar(im)

    for idx in range(len(result_items), nr*nc):
        grid[idx].set_visible(False)
        grid.cbar_axes[idx].set_visible(False)

    return fig


################################################################################
### Command-line interface

def add_bool_opt(parser, name, default=False):
    parser.add_argument('--' + name, action='store_true', default=default)
    parser.add_argument('--no-' + name, action='store_false',
            dest=name.replace('-', '_'))


def main():
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()

    parser.add_argument('results_file', metavar='FILE')
    parser.add_argument('keys', nargs='*',
            help="Choices: {}.".format(', '.join(sorted(KEY_NAMES))))

    add_bool_opt(parser, 'rmse', False)
    parser.add_argument('--cutoff', type=float, nargs='+', metavar='CUTOFF')
    add_bool_opt(parser, 'criteria', False)
    add_bool_opt(parser, 'criteria-firsts', False)
    add_bool_opt(parser, 'initial-preds', False)

    parser.add_argument('--all-plots', default=False, action='store_true')

    parser.add_argument('--cmap', default='jet')
    parser.add_argument('--outdir', nargs='?', const=True, default=None,
            metavar='DIR')
    add_bool_opt(parser, 'interactive', None)

    args = parser.parse_args()

    if args.all_plots:
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
            fig.savefig(fname, bbox_inches='tight', pad_inches=.1)
    else:
        def save_plot(name, fig=None):
            pass

    # interactive default depends on if outdir is present
    if args.interactive is None:
        args.interactive = not args.outdir

    # load the results
    with open(args.results_file, 'rb') as resultsfile:
        results = pickle.load(resultsfile)

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
    from matplotlib import cm
    cmap = cm.get_cmap(args.cmap)


    # RMSE plot
    if args.rmse:
        print("Plotting RMSEs")
        fig = plt.figure()
        plot_rmses(results)
        save_plot('rmses.png', fig)

    # plot of numbers >= a cutoff
    if args.cutoff is not None:
        for cutoff in args.cutoff:
            print("Plotting cutoff {}".format(cutoff))
            fig = plt.figure()
            plot_num_ge_cutoff(results, cutoff)
            save_plot('ge-{}.png'.format(cutoff), fig)

    # plot of each criterion
    if args.criteria:
        for criterion in args.keys:
            result = results[criterion]
            nice_name = KEY_NAMES[criterion]
            print("Plotting {}".format(nice_name))

            fig = plot_criteria_over_time(nice_name, result, cmap)
            save_plot('{}.png'.format(criterion), fig)

    # plot of criteria first steps
    if args.criteria_firsts:
        print("Plotting criteria first steps")
        items = sorted(
                ((k, v) for k, v in results.items() if k in args.keys),
                key=lambda item: KEY_NAMES[item[0]])

        fig = plot_criteria_firsts(items)
        save_plot('firsts.png', fig)

    # plot of initial predictions
    if args.initial_preds:
        print("Plotting initial predictions")
        fig = plt.figure()
        plot_predictions(results['_initial_apmf'], results['_real'])
        save_plot('initial_preds.png', fig)

    # pause to look at plots if interactive
    if args.interactive:
        plt.show()

if __name__ == '__main__':
    main()
