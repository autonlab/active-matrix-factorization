#!/usr/bin/env python3

import os
import pickle
import random

import numpy as np

import active_pmf
import bayes_pmf

import sys
sys.path.append('..')
import generate
import plot_results

def make_data_continuous(n, m, rank, u_mean=10, v_mean=10, noise=0):
    real, ratings, vals = active_pmf.make_fake_data(
            noise=noise, num_users=n, num_items=m, rank=rank,
            u_mean=u_mean, v_mean=v_mean,
    )
    return real

def make_data_discrete(n, m, rank):
    return generate.reconstruct(*
            generate.low_rank_approx(generate.make_orig(m, n), rank))

def make_known(n, m, num_known):
    known = generate.known_diag(n, m)

    if num_known == 'diag-minus-one':
        known[0,:-1] = 1
    else:
        unknown_indices = list((known == 0).reshape(-1).nonzero()[0])
        picked = random.sample(unknown_indices, num_known)
        known.flat[picked] = 1

    return known

def make_ratings(real, known):
    ratings = np.zeros((known.sum(), 3))
    for idx, (i, j) in enumerate(np.transpose(known.nonzero())):
        ratings[idx] = [i, j, real[i,j]]
    return ratings

def make_datafile(path, n, m, rank, num_known, kind='discrete', **make_args):
    if kind == 'discrete':
        real = make_data_discrete(n, m, rank)
        vals = generate.DEF_VALS
    elif kind == 'continuous':
        real = make_data_continuous(n, m, rank, **make_args)
        vals = None
    else:
        raise ValueError

    known = make_known(n, m, num_known)
    ratings = make_ratings(real, known)

    dct = {'_real': real, '_ratings': ratings, '_rating_vals': vals}

    with open(path, 'wb') as f:
        pickle.dump(dct, f)
    return dct

################################################################################

def get_apmf_criteria(data, save, latent_d, procs=None, refit_lookahead=True,
                      keys=None):
    if keys is None:
        keys = ['pred-variance']
        for x in ('pred-entropy-bound', 'uv-entropy', 'total-variance'):
            for y in ('', '-approx'):
                keys.append(x + y)

    real_ratings_vals = (data['_real'], data['_ratings'], data['_rating_vals'])

    results = active_pmf.compare(keys,
            real_ratings_vals=real_ratings_vals,
            latent_d=latent_d,
            discrete_exp=(data['_rating_vals'] is not None),
            refit_lookahead=refit_lookahead,
            fit_sigmas=False,
            steps=2,
            processes=procs, do_threading=True)

    if save:
        with open(save, 'wb') as f:
            pickle.dump(results, f)
    return results

def get_bayes_criteria(data, save, latent_d, procs=None, subtract_mean=False,
                       samps=100, lookahead_samps=100, keys=None):
    if keys is None:
        keys = ('pred-variance', 'exp-variance')

    results = bayes_pmf.compare_active(
            key_names=keys,
            latent_d=latent_d,
            real=data['_real'],
            ratings=data['_ratings'],
            rating_vals=data['_rating_vals'],
            num_steps=2,
            num_samps=samps, lookahead_samps=lookahead_samps,
            discrete=data['_rating_vals'] is not None,
            procs=procs, threaded=True)

    if save:
        with open(save, 'wb') as f:
            pickle.dump(results, f)
    return results

################################################################################


def plot(dirname, data, apmf_results, bayes_results, bayes_name='bayes.png'):
    def save_plot(name, fig=None):
        if fig is None:
            from matplotlib import pyplot as fig
        fname = os.path.join(dirname, name)
        fig.savefig(fname, bbox_inches='tight', pad_inches=.1)

    plot_results.plot_real(data['_real'], data['_ratings'])
    save_plot('real.png')

    if apmf_results:
        plot_results.plot_criteria_firsts(sorted(
            filter(lambda kv: not kv[0].startswith('_'), apmf_results.items()),
            key=lambda item: plot_results.KEY_NAMES[item[0]]))
        save_plot('apmf.png')

    if bayes_results:
        plot_results.plot_criteria_firsts(sorted(
            (('bayes_'+k,v) for k,v in bayes_results.items() if not k.startswith('_')),
            key=lambda item: plot_results.KEY_NAMES[item[0]]))
        save_plot(bayes_name)

################################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--rows', '-n', type=int, required=True)
    parser.add_argument('--cols', '-m', type=int, required=True)
    parser.add_argument('--rank', '-r', type=int, required=True)
    parser.add_argument('--latent-d', '-d', type=int, default=None)

    parser.add_argument('--num-known', '-K', required=True)

    parser.add_argument('--procs', '-p', type=int, default=None)

    parser.add_argument('--discrete', action='store_const', dest='type',
                        const='discrete', default='continuous')
    parser.add_argument('--continuous', action='store_const', dest='type',
                        const='continuous')

    parser.add_argument('--u-mean', type=float, default=10)
    parser.add_argument('--v-mean', type=float, default=10)
    parser.add_argument('--noise', type=float, default=0)

    parser.add_argument('--samps', type=int, default=100)
    parser.add_argument('--lookahead-samps', type=int, default=100)

    parser.add_argument('--refit-lookahead', action='store_true', default=True)
    parser.add_argument('--no-refit-lookahead', action='store_false',
                        dest='refit_lookahead')

    parser.add_argument('--no-apmf', action='store_false', default=True,
                        dest='do_apmf')
    parser.add_argument('--no-bayes', action='store_false', default=True,
                        dest='do_bayes')
    parser.add_argument('--no-plot', action='store_false', default=True,
                        dest='do_plot')

    parser.add_argument('dir')
    args = parser.parse_args()

    
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if args.latent_d is None:
        args.latent_d = args.rank

    try:
        args.num_known = int(args.num_known)
    except ValueError:
        pass

    
    datapath = os.path.join(args.dir, 'data.pkl')
    if os.path.exists(datapath):
        print("Using existing data: {}".format(datapath))
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
    else:
        data = make_datafile(datapath,
                args.rows, args.cols, args.rank, args.num_known, args.type,
                u_mean=args.u_mean, v_mean=args.v_mean, noise=args.noise)

    if args.do_apmf:
        print()
        print('=' * 80)
        print()

        apmf_path = os.path.join(args.dir, 'apmf.pkl')
        if os.path.exists(apmf_path):
            print("APMF results already exist: {}".format(apmf_path))
            with open(apmf_path, 'rb') as f:
                apmf_results = pickle.load(f)
        else:
            apmf_results = get_apmf_criteria(data, apmf_path, args.latent_d,
                    args.procs, args.refit_lookahead)
    else:
        apmf_results = {}

    if args.do_bayes:
        print()
        print('=' * 80)
        print()

        bayes_name = 'bayes_{}_{}'.format(args.samps, args.lookahead_samps)
        bayes_path = os.path.join(args.dir, bayes_name + '.pkl')
        if os.path.exists(bayes_path):
            print("Bayes results already exist: {}".format(bayes_path))
            with open(bayes_path, 'rb') as f:
                bayes_results = pickle.load(f)
        else:
            bayes_results = get_bayes_criteria(data, bayes_path,
                    args.latent_d, args.procs,
                    samps=args.samps, lookahead_samps=args.lookahead_samps)
    else:
        bayes_results = {}

    if args.do_plot:
        print()
        print('=' * 80)
        print()

        import matplotlib
        matplotlib.use('Agg')
        plot(args.dir, data, apmf_results, bayes_results, bayes_name + '.png')

if __name__ == '__main__':
    main()
