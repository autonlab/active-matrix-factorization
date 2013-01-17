#!/usr/bin/env python
'''
Implementation of Bayesian PMF with Hamiltonian MCMC/NUTS (through Stan).
'''

# TODO: don't pickle big things for pool, inherit them: http://goo.gl/tpS15
# TODO: save in more portable format than pickle?

from __future__ import print_function, division

from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import partial
from itertools import islice, product, repeat
import multiprocessing
import random
from threading import Thread
import warnings

import numpy as np
from scipy import stats
import six
import six.moves as sixm

# XXX this import isn't in this folder...
# from pmf_cy import ProbabilisticMatrixFactorization, rmse, parse_fit_type

from rstan_interface import get_model, sample
stan_model = get_model('bpmf.stan')


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).sum() / a.size)


class BayesianPMF(object):
    def __init__(self, rating_tuples, latent_d,
                 subtract_mean=True,
                 rating_values=None,
                 discrete_expectations=True,
                 num_integration_pts=50,
                 knowable=None):
        self.latent_d = latent_d
        self.subtract_mean = subtract_mean

        self.rating_std = 2
        self.mu_0 = np.zeros(latent_d)  # mean for future means

        # feature mean covariances are beta_0 * inv wishart(nu_0, w_0)
        self.beta_0 = 2  # observation noise precision
        self.nu_0 = latent_d  # degrees of freedom
        self.w_0 = np.eye(latent_d)  # scale matrix, PSD

        self.ratings = np.array(rating_tuples, dtype=float, copy=False)
        if self.ratings.shape[1] != 3:
            raise TypeError("invalid rating tuple length")
        self.mean_rating = np.mean(self.ratings[:, 2])

        self.num_users = n = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = m = int(np.max(self.ratings[:, 1]) + 1)

        self.rated = set((i, j) for i, j, rating in self.ratings)
        if knowable is None:
            knowable = product(sixm.xrange(n), sixm.xrange(m))
        self.unrated = set(knowable).difference(self.rated)

        if rating_values is not None:
            rating_values = set(sixm.map(float, rating_values))
            if not rating_values.issuperset(self.ratings[:, 2]):
                raise ValueError("got ratings not in rating_values")
        self.rating_values = rating_values
        self.discrete_expectations = discrete_expectations
        self.num_integration_pts = num_integration_pts

        # keep track of the highest-likelihood sample so far
        # we'll sometimes want to initialize sampling here
        self.sampled_mode = None
        self.sampled_mode_lp = -np.inf

    def _set_rating_values(self, vals):
        if vals:
            vals = tuple(sorted(vals))
            self._rating_values = vals

            varray = np.empty(len(vals) + 2)
            varray[0] = -np.inf
            varray[1:-1] = vals
            varray[-1] = np.inf

            self._rating_bounds = (varray[1:] + varray[:-1]) / 2
        else:
            self._rating_values = None
            self._rating_bounds = None

    rating_values = property(lambda self: self._rating_values,
                             _set_rating_values)
    rating_bounds = property(lambda self: self._rating_bounds)

    def add_rating(self, i, j, rating):
        self.add_ratings([i, j, rating])

    def add_ratings(self, extra):
        rows, cols = self.ratings.shape

        extra = np.array(extra, copy=False, ndmin=2)
        if len(extra.shape) != 2 or extra.shape[1] != cols:
            raise TypeError("bad shape for extra")

        assert np.max(extra[:, 0] + 1) <= self.num_users
        assert np.max(extra[:, 1] + 1) <= self.num_items

        rating_values = getattr(self, 'rating_values', None)
        if rating_values is not None:
            if not set(rating_values).issuperset(extra[:, 2]):
                raise ValueError("got ratings with bad values")

        new_items = set((int(i), int(j)) for i, j in extra[:, :2])

        if not new_items.isdisjoint(self.rated):
            raise ValueError("can't rate already rated items")
        self.rated.update(new_items)
        self.unrated.difference_update(new_items)

        self.ratings = np.append(self.ratings, extra, 0)
        self.mean_rating = np.mean(self.ratings[:, 2])
        # TODO: this can be done without a copy by .resize()...

        # keep the old sampled mode, but immediately replace it once we have
        # a new one, since the old log-likelihood is no longer valid
        self.sampled_mode_lp = -np.inf

    def samples(self, num_samps, warmup=None, chains=1,
                start_at_mode=True, update_mode=True):
        '''
        Runs the Markav chain for num_samps samples, after warming up for
        warmup iterations beforehand (default: num_samps // 2).

        Returns a dictionary with keys the name of the param, values
        an array with first dimension num_samps.
        '''
        # TODO: options about starting at previous points...
        if warmup is None:
            warmup = num_samps // 2

        data = {
            'n_users': self.num_users,
            'n_items': self.num_items,
            'rank': self.latent_d,

            'n_obs': self.ratings.shape[0],
            'obs_users': self.ratings[:, 0] + 1,
            'obs_items': self.ratings[:, 1] + 1,
            'obs_ratings': self.ratings[:, 2],

            'rating_std': self.rating_std,
            'mu_0': self.mu_0,
            'beta_0': self.beta_0,
            'nu_0': self.nu_0,
            'w_0': self.w_0,
        }
        args = {'chains': chains, 'iter': warmup + num_samps, 'warmup': warmup,
                'eat_output': True, 'return_output': False}
        if start_at_mode:
            args['init'] = self.sampled_mode
        samples = sample(stan_model, data=data, **args)

        if update_mode:
            i = samples['lp__'].argmax()
            if samples['lp__'][i] > self.sampled_mode_lp:
                self.sampled_mode = {k: v[i] for k, v in six.iteritems(samples)}
                self.sampled_mode_lp = samples['lp__'][i]
        return samples

    def matrix_results(self, vals, which):
        "Returns a num_users x num_items matrix with `which` set to `vals`."
        res = np.empty((self.num_users, self.num_items))
        res.fill(np.nan)
        res[which] = vals
        return res

    def pick_out_predictions(self, samples, which=Ellipsis):
        # TODO: better way to index with which on the non-first axis
        return np.asarray([p[which] for p in samples['predictions']])

    def predict(self, samples, which=Ellipsis):
        "Gives the mean reconstruction given a series of samples."
        return np.mean(self.pick_out_predictions(samples, which), axis=0)

    def pred_variance(self, samples, which=Ellipsis):
        "Gives the variance of each prediction in a series of samples."
        return np.var(self.pick_out_predictions(samples, which), axis=0)

    def total_variance(self, samples, which=Ellipsis):
        '''
        Gives the sum of the variance of each element of the prediction matrix
        selected by which over samples_iter: \sum Var[R_ij].
        '''
        return self.pred_variance(samples, which=which).sum()

    def exp_variance(self, samples, which=Ellipsis, pool=None,
                     num_samps=30, warmup=15, **sample_args):
        '''
        Gives the total expected variance in our predicted R matrix if we knew
        Rij for each ij in which, using samples to represent our distribution
        for Rij: \sum E[Var[R_ij]].

        Parallelizes the evaluation over pool if passed (highly recommended,
        since this is sloooow).
        '''
        return self._distribute(_exp_variance_helper,
                samples=samples, which=which, pool=pool,
                num_samps=num_samps, warmup=warmup, **sample_args)

    def _distribute(self, fn, samples, which, pool, **sample_args):
        # figure out the indices of our selection
        # ...there's gotta be an easier way, right?
        n = self.num_users
        m = self.num_items
        all_indices = np.empty((n, m, 2), dtype=int)
        for i in sixm.xrange(n):
            all_indices[i, :, 0] = i
        for j in sixm.xrange(m):
            all_indices[:, j, 1] = j
        indices = all_indices[which]
        i_indices = indices[..., 0]
        j_indices = indices[..., 1]

        # get samples of R_ij for each ij in which
        vals = self.pick_out_predictions(samples, which)

        # estimate the marginal distribution of each R_ij from the samples
        if self.discrete_expectations and self.rating_values is not None:
            discrete = True

            # round the samples to the nearest thing in rating_vals
            # and do a MAP fit to a categorical with a Dirichlet prior alpha+1
            alpha = .1
            prev_samps = vals.shape[0]
            denom = prev_samps + alpha * len(self.rating_values)
            params = [
                (np.histogram(v, bins=self.rating_bounds)[0] + alpha) / denom
                for v in vals.reshape(prev_samps, -1).T
            ]

        else:
            if self.discrete_expectations and self.rating_values is None:
                warnings.warn("have no rating_values; doing continuous")
            discrete = False

            # fit an MLE normal to the samples
            mean = np.mean(vals, 0)
            var = np.var(vals, 0)
            params = sixm.zip(mean.flat, var.flat)

        # TODO could experiment with map_async, imap_unordered
        mapper = pool.map if pool is not None else map
        exps = mapper(partial(fn, **sample_args),
                sixm.zip(repeat(self),
                    i_indices.flat, j_indices.flat, repeat(discrete), params))

        res = np.empty(np.shape(vals)[1:])
        res.fill(np.nan)
        for idx, exp in enumerate(exps):
            res.flat[idx] = exp
        return res

    def prob_ge_cutoff(self, samples, cutoff, which=Ellipsis):
        '''
        Gives the portion of the time each matrix element was >= cutoff
        in a series of samples.
        '''
        preds = self.pick_out_predictions(samples, which)
        return np.mean(preds >= cutoff, axis=0)

    def random(self, samples, which=Ellipsis):
        "Random scoring."
        shape = np.empty((self.num_users, self.num_items))[which].shape
        return np.random.rand(*shape)

    def bayes_rmse(self, samples, true_r, which=Ellipsis):
        "The RMSE of predictions compared to true values."
        return rmse(self.predict(samples, which), true_r[which])


# stupid functions to work around multiprocessing.Pool/pickle silliness

def _integrate_lookahead(fn, bpmf, i, j, discrete, params, **sample_args):
    if (i, j) in bpmf.rated:
        warnings.warn("Asked to check a known entry; returning NaN")
        return np.nan

    def calculate_fn(v):
        b = deepcopy(bpmf)
        b.add_rating(i, j, v)
        samps = b.samples(**sample_args)
        return fn(b, samps)

    if discrete:
        # TODO: trapezoidal approximation? simpsons?
        # TODO: don't calculate for points with very low probability?
        evals = np.array([calculate_fn(v) for v in bpmf.rating_values])
        est = (evals * params).sum()
        s = "summed"
    else:
        mean, var = params
        dist = stats.norm(loc=mean, scale=np.sqrt(var))

        # find points to evaluate at
        # TODO: do fewer evaluations if the distribution is really narrow
        pts = dist.ppf(np.linspace(.001, .999, bpmf.num_integration_pts))
        evals = np.fromiter(sixm.map(calculate_fn, pts), float, pts.size)
        est = np.trapz(evals * dist.pdf(pts), pts)
        s = "from {:5.1f} to {:5.1f} in {}".format(pts[0], pts[-1], pts.size)

        # # only take the expectation out to 2 sigma (>95% of normal mass)
        # left = mean - 2 * std
        # right = mean + 2 * std
        # est = stats.norm.expect(calculate_fn, loc=mean, scale=np.sqrt(var),
        #         lb=left, ub=right, epsrel=.05) # XXX integration eps
        # s = "from {:5.1f} to {:5.1f}".format(left, right)

    name = getattr(fn, '__name__', '')
    print("\t{:>20}({},{}) {}: {: 10.2f}".format(name, i, j, s, est))
    return est


def _exp_variance_helper(args, **sample_args):
    return _integrate_lookahead(BayesianPMF.total_variance, *args, **sample_args)

################################################################################

# The various evaluation functions we consider
Key = namedtuple('Key', 'nice_name key_fn choose_max does_sampling args')
KEYS = {
    'random': Key("Random", 'random', True, False, ()),
    'pred-variance': Key("Var[R_ij]", 'pred_variance', True, False, ()),

    'exp-variance': Key("E[Var[R]]", 'exp_variance', False, True, ()),

    'pred': Key("Pred", 'predict', True, False, ()),
    'prob-ge-3.5': Key("Prob >= 3.5", 'prob_ge_cutoff', True, False, (3.5,)),
    'prob-ge-.5': Key("Prob >= .5", 'prob_ge_cutoff', True, False, (.5,)),
    'prob-ge-0': Key("Prob >= 0", 'prob_ge_cutoff', True, False, (0,)),
}


def fetch_samples(bpmf, num_samps, **kwargs):
    try:
        samps = bpmf.samples(num_samps=num_samps, **kwargs)
        pred = bpmf.predict(samps)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    return samps, pred


def full_test(bpmf, samples, real, key_name,
              num_samps=128, samp_args=None,
              lookahead_samps=128, lookahead_samp_args=None,
              pool=None, sample_in_pool=False, test_on=Ellipsis):
    '''
    Evaluates the chosen selection criterion (key_name) on the data (real),
    starting with an initial instance of BayesianPMF and some samples from it.

    Yields tuples of # of rated items, the current RMSE, the (i, j) index of
    the last choice, and an array of evaluations for the full matrix.
    '''
    key = KEYS[key_name]
    total = real.size
    picker_fn = getattr(bpmf, key.key_fn)
    chooser = np.argmax if key.choose_max else np.argmin

    samp_args = (samp_args or {}).copy()
    samp_args['num_samps'] = num_samps

    lookahead_samp_args = (lookahead_samp_args or {}).copy()
    lookahead_samp_args['num_samps'] = lookahead_samps

    init_rmse = bpmf.bayes_rmse(samples, real, which=test_on)
    yield (len(bpmf.rated), init_rmse, None, None)

    while bpmf.unrated:
        print("{:<40} Picking query point {}...".format(
            key.nice_name, len(bpmf.rated) + 1))

        if len(bpmf.unrated) == 1:
            vals = None
            i, j = next(iter(bpmf.unrated))
        else:
            unrated = np.array(list(bpmf.unrated)).T
            which = tuple(unrated)

            key_kwargs = {'which': which}
            if key.does_sampling:
                key_kwargs.update(lookahead_samp_args)
                if pool is not None:
                    key_kwargs['pool'] = pool

            evals = picker_fn(samples, *key.args, **key_kwargs)

            i, j = unrated[:, chooser(evals)]
            vals = bpmf.matrix_results(evals, which)

        bpmf.add_rating(i, j, real[i, j])
        print("{:<40} Queried ({}, {}); {}/{} known".format(
                key.nice_name, i, j, len(bpmf.rated), total))

        if sample_in_pool:
            samples, pred = pool.apply(fetch_samples, [bpmf], samp_args)
        else:
            samples, pred = fetch_samples(bpmf, **samp_args)

        err = rmse(pred[test_on], real[test_on])
        print("{:<40} RMSE {}: {:.5}".format(
            key.nice_name, len(bpmf.rated), err))
        yield len(bpmf.rated), err, (i, j), vals


def compare_active(key_names, latent_d, real, ratings, rating_vals=None,
                   discrete=True, subtract_mean=True, num_integration_pts=50,
                   num_steps=None, procs=None, threaded=False,
                   num_samps=128, samp_args=None, test_set='all', **kwargs):
    # figure out which points we know the answers to
    knowable = np.isfinite(real)
    knowable[real == 0] = 0

    # ... and which points we can query
    pickable = knowable.copy()
    pickable[ratings[:, 0].astype(int), ratings[:, 1].astype(int)] = 0

    # figure out test set
    try:
        test_set = float(test_set)
    except ValueError:
        if test_set != 'all':
            warnings.warn("dunno what to do with test_set {}".format(test_set))
            test_set = 'all'

    if test_set == 'all':
        test_on = knowable
        query_on = pickable
    else:
        if test_set % 1 == 0 and test_set != 1:
            avail_pts = list(sixm.zip(*pickable.nonzero()))
            picked_indices = random.sample(avail_pts, int(test_set))
            picker = np.zeros(pickable.shape, bool)
            picker[tuple(np.transpose(picked_indices))] = 1
        else:
            picker = np.random.binomial(1, test_set, size=pickable.shape)
        test_on = picker * pickable
        query_on = (1 - picker) * pickable

    query_set = set(sixm.zip(*query_on.nonzero()))

    print("{} points known, {} to query, testing on {}, {} knowable, {} total"
            .format(ratings.shape[0], query_on.sum(), test_on.sum(),
                    knowable.sum(), real.size))

    bpmf_init = BayesianPMF(ratings, latent_d,
            subtract_mean=subtract_mean,
            rating_values=rating_vals,
            discrete_expectations=discrete,
            num_integration_pts=num_integration_pts,
            knowable=query_set)

    pool = multiprocessing.Pool(procs) if procs is None or procs > 1 else None

    print("Getting initial MCMC samples...")
    samples = bpmf_init.samples(num_samps=num_samps, **samp_args)

    init_rmse = bpmf_init.bayes_rmse(samples, real, test_on)
    print("Initial RMSE: {}".format(init_rmse))
    print()

    results = {
        '_real': real,
        '_ratings': ratings,
        '_rating_vals': rating_vals,
        '_initial_bpmf': deepcopy(bpmf_init),
    }

    # continue with each key for the fit
    def eval_key(key_name):
        res = full_test(
                deepcopy(bpmf_init), samples, real, key_name,
                num_samps=num_samps, samp_args=samp_args,
                pool=pool, sample_in_pool=threaded, test_on=test_on, **kwargs)
        results[key_name] = list(islice(res, num_steps))

    # TODO: no concurrent access to R. be careful here...
    if threaded:
        threads = [Thread(name=key_name, target=eval_key, args=(key_name,))
                   for key_name in key_names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    else:
        for key_name in key_names:
            eval_key(key_name)

    if pool is not None:
        pool.close()
        pool.join()

    return results


def main():
    import argparse
    import os
    import pickle
    import sys

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

    key_names = KEYS.keys()

    # set up arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--latent-d', '-D', type=int, default=5)
    parser.add_argument('--steps', '-s', type=int, default=None)

    parser._add_action(ActionNoYes('discrete', default=None))
    parser.add_argument('--num-integration-pts', type=int, default=50)

    parser._add_action(ActionNoYes('subtract-mean', default=True))

    parser.add_argument('--samps', '-S', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=50)

    parser.add_argument('--lookahead-samps', type=int, default=100)
    parser.add_argument('--lookahead-warmup', type=int, default=50)

    parser._add_action(ActionNoYes('threaded', 'unthreaded', default=True))
    parser.add_argument('--procs', '-P', type=int, default=None)

    parser.add_argument('--test-set', default='all')

    parser.add_argument('--load-data', required='True', metavar='FILE')
    parser.add_argument('--save-results', nargs='?', default=True, const=True,
            metavar='FILE')
    parser.add_argument('--no-save-results',
            action='store_false', dest='save_results')

    parser.add_argument('--notes', nargs='+',
        help="Doesn't do anything, just there to save any notes you'd like "
             "in the results file.")

    parser.add_argument('keys', nargs='*',
            help="Choices: {}.".format(', '.join(sorted(key_names))))

    args = parser.parse_args()

    # check that args.keys are valid
    for k in args.keys:
        if k not in key_names:
            sys.stderr.write("Invalid key name %s; options are %s.\n" % (
                k, ', '.join(sorted(key_names))))
            sys.exit(1)

    if not args.keys:
        args.keys = sorted(key_names)

    # make directories to save results if necessary
    if args.save_results is True:
        args.save_results = 'results.pkl'
    elif args.save_results:
        dirname = os.path.dirname(args.save_results)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

    # load data
    with open(args.load_data, 'rb') as f:
        data = np.load(f)

        if isinstance(data, np.ndarray):
            data = {'_real': data}

        real = data['_real']
        ratings = data['_ratings']
        rating_vals = data['_rating_vals'] if '_rating_vals' in data else None

    if args.discrete is None:
        args.discrete = rating_vals is not None

    # do the comparison
    try:
        results = compare_active(
                key_names=args.keys,
                latent_d=args.latent_d,
                real=real, ratings=ratings, rating_vals=rating_vals,
                test_set=args.test_set, num_steps=args.steps,
                subtract_mean=args.subtract_mean,
                discrete=args.discrete,
                num_integration_pts=args.num_integration_pts,
                num_samps=args.samps, lookahead_samps=args.lookahead_samps,
                samp_args={'warmup': args.warmup},
                lookahead_samp_args={'warmup': args.lookahead_warmup},
                procs=args.procs, threaded=args.threaded)
    except Exception:
        import traceback
        print()
        traceback.print_exc()

        import pdb
        print()
        pdb.post_mortem()

        sys.exit(1)

    # save the results file
    if args.save_results:
        print("\nsaving results in '{}'".format(args.save_results))

        results['_args'] = args

        with open(args.save_results, 'wb') as f:
            pickle.dump(results, f, protocol=2)

if __name__ == '__main__':
    main()
