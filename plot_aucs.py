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
                         do_cutoffs=None, do_cutoff_aucs=None,
                         ret_rmse_traces=False, ret_cutoff_traces=False,
                         rmse_over_random=False):
    desired_ns = None

    want_rmses = do_rmse or do_rmse_auc or ret_rmse_traces
    if want_rmses:
        rmse_traces = defaultdict(list)

    cutoff_vals = set()
    if do_cutoffs:
        cutoff_vals.update(do_cutoffs)
    if do_cutoff_aucs:
        cutoff_vals.update(do_cutoff_aucs)

    if cutoff_vals or ret_cutoff_traces:
        cutoff_traces = defaultdict(functools.partial(defaultdict, list))

    if not want_rmses and not cutoff_vals:  # not asked to do anything!
        return {}

    for r in map(load_results, filenames):
        if cutoff_vals:
            real = r['_real']
            ratings = r['_ratings']

        if want_rmses and rmse_over_random:
            random = [(k, v) for k, v in r.items() if k.endswith('random')]
            assert len(random) == 1
            random_rmse = np.asarray([r[1] for r in random[0][1]])

        for k, v in r.items():
            if k.startswith('_'):
                continue

            ns, rmses, ijs, evals = zip(*v)
            ns = np.asarray(ns)
            rmses = np.asarray(rmses)
            if desired_ns is not None:
                assert np.all(ns == desired_ns)
            else:
                desired_ns = ns

            if want_rmses:
                if rmse_over_random:
                    rmses -= random_rmse
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

    if not ret_rmse_traces and not ret_cutoff_traces:
        return results
    else:
        ret = [results]
        if ret_rmse_traces:
            ret.append({k: np.asarray(v) for k, v in rmse_traces.items()})
        if ret_cutoff_traces:
            ret.append({k: np.asarray(v) for k, v in cutoff_traces.items()})
        return ret


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

    plt.tight_layout()


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
            # jiggle anything that's all exactly one value, to avoid singularity
            aucs = [grp if len(set(grp)) > 1
                        else list(grp) + [grp[0] + .01]
                    for grp in aucs]
            beanplot(aucs, ax=plt.gca(), plot_opts={'cutoff': True})
        indices = np.arange(len(names)) + 1
    plt.xticks(indices, names, rotation=90)
    plt.xlim(indices[0] - .5, indices[-1] + .5)
    plt.hlines(0, *plt.xlim(), color='k')
    plt.ylabel(ylabel)
    plt.tight_layout()

################################################################################


def main():
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
    g.add_argument('--ge-cutoff', nargs='+', type=float)
    g.add_argument('--ge-cutoff-auc', nargs='+', type=float)

    #parser.add_argument('--save')
    args = parser.parse_args()

    #if args.save:
    #    import matplotlib
    #    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    res = load_data(args.files,
        do_rmse=args.rmses, do_rmse_auc=args.auc,
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

    rmse_name = 'RMSE over random' if args.over_random else 'RMSE'

    if args.rmses:
        plt.figure()
        plot_lines(ns, filter_keys(data['rmse']), rmse_name)
        show_legend(args.legend)

    if args.rmse_fboxplots:
        from statsmodels.graphics.functional import fboxplot
        for name, trace in filter_keys(rmse_traces).items():
            fboxplot(trace, xdata=ns)
            plt.hlines(0, *plt.xlim(), color='k')
            plt.title(KEY_NAMES.get(name, name))
            plt.xlabel("# of rated elements")
            plt.ylabel(rmse_name)

    if args.auc:
        plt.figure()
        plot_aucs(filter_keys(data['rmse_auc']), 'AUC ({})'.format(rmse_name))

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
