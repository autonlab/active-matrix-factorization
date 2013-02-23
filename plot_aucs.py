#!/usr/bin/env python3

from collections import defaultdict
import functools
import itertools
import math
import re
import sys

import numpy as np

from plot_results import (KEY_NAMES,
    ActivePMF, MNActivePMF, BayesianPMF, BPMF, NewItemsBPMF,  # for pickle
    linestyle_color_marker, load_results, auc_roc)

################################################################################
### loading data

def load_data(filenames, do_rmse=False, do_rmse_auc=False,
                         do_predauc=False, do_predauc_auc=False,
                         do_cutoffs=None, do_cutoff_aucs=None,
                         ret_rmse_traces=False, ret_cutoff_traces=False,
                         ret_predauc_traces=False,
                         rmse_over_random=False,
                         rmse_div_random=False,
                         already_loaded=False):
    desired_ns = None

    assert not rmse_over_random or not rmse_div_random

    want_rmses = do_rmse or do_rmse_auc or ret_rmse_traces
    if want_rmses:
        rmse_traces = defaultdict(list)

    want_predaucs = do_predauc or do_predauc_auc or ret_predauc_traces
    if want_predaucs:
        predauc_traces = defaultdict(list)

    cutoff_vals = set()
    if do_cutoffs:
        cutoff_vals.update(do_cutoffs)
    if do_cutoff_aucs:
        cutoff_vals.update(do_cutoff_aucs)

    if cutoff_vals or ret_cutoff_traces:
        cutoff_traces = defaultdict(functools.partial(defaultdict, list))

    if not want_rmses and not cutoff_vals:  # not asked to do anything!
        return {}

    for r in (filenames if already_loaded else map(load_results, filenames)):
        if cutoff_vals:
            real = r['_real']
            ratings = r['_ratings']

        if want_predaucs:
            test_on = r['_test_on']
            label = r['_real'][test_on] > 0

        if rmse_over_random or rmse_div_random:
            random, = [v for k, v in r.items() if k.endswith('random')]
            if want_rmses:
                random_rmse = np.asarray([r[1] for r in random])
                if rmse_div_random:
                    random_rmse_finite = np.isfinite(random_rmse)
            if want_predaucs:
                random_predauc = np.asarray([
                    auc_roc(r[4][test_on], label)[0]
                    if len(r) >= 5
                    else np.nan
                    for r in random
                ])
                if rmse_predauc_random:
                    random_predauc_finite = np.isfinite(random_predauc)

        for k, v in r.items():
            if k.startswith('_'):
                continue

            if len(v[0]) == 4:
                ns, rmses, ijs, evals = zip(*v)
            else:
                ns, rmses, ijs, evals, preds = zip(*v)

            ns = np.asarray(ns)
            rmses = np.asarray(rmses)
            if desired_ns is not None:
                assert np.all(ns == desired_ns)
            else:
                desired_ns = ns

            if want_rmses:
                if rmse_over_random:
                    rmses -= random_rmse
                elif rmse_div_random:
                    rmses[random_rmse_finite] /= random_rmse[random_rmse_finite]
                rmse_traces[k].append(rmses)

            if want_predaucs:
                predaucs = np.array([
                    np.nan if pred is None else auc_roc(pred[test_on], label)[0]
                    for pred in preds
                ])
                if rmse_over_random:
                    predaucs -= random_predauc
                if rmse_div_random:
                    predaucs[random_predauc_finite] /= random_predauc[random_predauc_finite]
                predauc_traces[k].append(predaucs)

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

    if do_predauc:  # name => mean predauc curve
        results['predauc'] = {
            k: np.mean(v, axis=0) for k, v in predauc_traces.items()
        }

    if do_predauc_auc:  # name => array of area under RMSE curves
        results['predauc_auc'] = {
            k: np.trapz(v, axis=1) for k, v in predauc_traces.items()
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

    if ret_rmse_traces or ret_predauc_traces or ret_cutoff_traces:
        ret = [results]
        if ret_rmse_traces:
            ret.append({k: np.asarray(v) for k, v in rmse_traces.items()})
        if ret_cutoff_traces:
            ret.append({k: np.asarray(v) for k, v in cutoff_traces.items()})
        if ret_predauc_traces:
            ret.append({k: np.asarray(v) for k, v in predauc_traces.items()})
        return ret
    else:
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


def plot_lines(ns, data, ylabel=None, names=None, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    ax.set_xlabel("# of rated elements")
    if ylabel:
        ax.set_ylabel(ylabel)

    if names is None:
        names = KEY_NAMES

    nice_results = sorted(
            ((names[k], v) for k, v in data.items()),
            key=lambda kv: np.trapz(kv[1]), reverse=True)

    # offset lines a bit so you can see when some of them overlap
    total = len(ns)
    offset = .15 / total if total > 3 else .02

    # cycle through colors and line styles
    l_c_m = linestyle_color_marker(len(ns))

    for idx, (nice_name, vals) in enumerate(nice_results):
        nums = ns + (idx - total / 2) * offset

        l, c, m = next(l_c_m)
        ax.plot(nums, vals, linestyle=l, color=c, label=nice_name, marker=m)

    # only show integer values for x ticks
    #xmin, xmax = plt.xlim()
    #plt.xticks(range(math.ceil(xmin), math.floor(xmax) + 1))

    #plt.tight_layout()


def plot_aucs(aucs, ylabel=None, names=None, rotation=90, ha='center'):
    import matplotlib.pyplot as plt
    if names is None:
        names = KEY_NAMES
    names, aucs = list(zip(*sorted((names[k], v) for k, v in aucs.items())))

    if all(a.size == 1 for a in aucs):
        plt.plot(aucs, linestyle='None', marker='o')
        indices = np.arange(len(names))
    else:
        try:
            from statsmodels.graphics.boxplots import beanplot
        except ImportError:
            plt.boxplot(aucs)
        else:
            # jiggle anything that's all exactly one value, to avoid singularity
            aucs = [grp if len(set(grp)) > 1
                        else list(grp) + [grp[0] + .01]
                    for grp in aucs]
            beanplot(aucs, ax=plt.gca(), plot_opts={'cutoff': True})
        indices = np.arange(len(names)) + 1
    plt.xticks(indices, names, rotation=rotation, ha=ha)
    plt.xlim(indices[0] - .5, indices[-1] + .5)

    bot, top = plt.ylim()
    if bot < 0 < top:
        plt.hlines(0, *plt.xlim(), color='k')
    if ylabel:
        plt.ylabel(ylabel)

################################################################################


def main(argstr=None):
    import argparse

    # helper for boolean flags
    # based on http://stackoverflow.com/a/9236426/344821
    class ActionNoYes(argparse.Action):
        def __init__(self, opt_name, off_name=None, dest=None,
                     default=True, required=False, help=None):

            if off_name is None:
                off_name = 'no-' + opt_name
            self.off_name = '--' + off_name

            if dest is None:
                dest = opt_name.replace('-', '_')

            super(ActionNoYes, self).__init__(
                    ['--' + opt_name, '--' + off_name],
                    dest, nargs=0, const=None,
                    default=default, required=required, help=help)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, option_string != self.off_name)

    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--over-random', action='store_true', default=False)
    g.add_argument('--absolute', action='store_false', dest='over_random')

    parser.add_argument('--key-regexes', '--keys', nargs='*', metavar='RE',
                        default=[re.compile('.*')], type=re.compile)
    parser.add_argument('--key-exclude-regexes', '--skip-keys', nargs='*',
                        default=[], type=re.compile, metavar='RE')

    parser.add_argument('--legend', default='outside',
                        choices={'outside', 'inside'})

    g = parser.add_argument_group('Plot Types')
    g._add_action(ActionNoYes('rmses', default=False))
    g._add_action(ActionNoYes('rmse-fboxplots', default=False))
    g._add_action(ActionNoYes('auc', default=True))
    g._add_action(ActionNoYes('predaucs', default=False))
    g._add_action(ActionNoYes('predauc-auc', default=False))
    g.add_argument('--ge-cutoff', nargs='+', type=float)
    g.add_argument('--ge-cutoff-auc', nargs='+', type=float)

    #parser.add_argument('--save')
    if argstr is not None:
        import shlex
        args = parser.parse_args(shlex.split(argstr))
    else:
        args = parser.parse_args()

    #if args.save:
    #    import matplotlib
    #    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    res = load_data(args.files,
        do_rmse=args.rmses, do_rmse_auc=args.auc,
        do_predauc=args.predaucs, do_predauc_auc=args.predauc_auc,
        do_cutoffs=args.ge_cutoff, do_cutoff_aucs=args.ge_cutoff_auc,
        rmse_over_random=args.over_random,
        ret_rmse_traces=args.rmse_fboxplots)
    if args.rmse_fboxplots:
        data, rmse_traces = res
    else:
        data = res
    ns = data['ns']

    #key_res = [re.compile(r) for r in args.key_regexes]
    #key_bads = [re.compile(r) for r in args.key_exclude_regexes]
    def filter_keys(d):
        return {k: v for k, v in d.items()
                if any(r.search(k) for r in args.key_regexes)
                and not any(r.search(k) for r in args.key_exclude_regexes)}

    over_random = ' over_random' if args.over_random else ''

    # rmse stuff
    if args.rmses:
        plt.figure()
        plot_lines(ns, filter_keys(data['rmse']), 'RMSE' + over_random)
        show_legend(args.legend)

    if args.rmse_fboxplots:
        from statsmodels.graphics.functional import fboxplot
        for name, trace in filter_keys(rmse_traces).items():
            fboxplot(trace, xdata=ns)
            plt.hlines(0, *plt.xlim(), color='k')
            plt.title(KEY_NAMES.get(name, name))
            plt.xlabel("# of rated elements")
            plt.ylabel('RMSE' + over_random)

    if args.auc:
        plt.figure()
        plot_aucs(filter_keys(data['rmse_auc']),
                  'AUC ({})'.format('RMSE' + over_random))

    # prediction auc stuff
    if args.predaucs:
        plt.figure()
        plot_lines(ns, filter_keys(data['predauc']),
                   'Prediction AUC' + over_random)
        show_legend(args.legend)

    if args.auc:
        plt.figure()
        plot_aucs(filter_keys(data['predauc_auc']),
                  'AUC ({})'.format('Prediction AUC' + over_random))

    # # >= cutoff stuff
    if args.ge_cutoff:
        for cutoff in args.ge_cutoff:
            plt.figure()
            plot_lines(ns, filter_keys(data['cutoffs'][cutoff]),
                       '# >= {}'.format(cutoff))
            show_legend(args.legend)

    if args.ge_cutoff_auc:
        for cutoff in args.ge_cutoff_auc:
            plt.figure()
            plot_aucs(filter_keys(data['cutoff_aucs'][cutoff]),
                      'AUC (# >= {})'.format(cutoff))

    plt.show()

if __name__ == '__main__':
    main()
