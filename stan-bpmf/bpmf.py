#!/usr/bin/env python
'''
Implementation of Bayesian PMF with Hamiltonian MCMC/NUTS (through Stan).
'''

# TODO: don't pickle big things for pool, inherit them: http://goo.gl/tpS15
# TODO: save in more portable format than pickle?

from __future__ import print_function, division

from collections import namedtuple, Counter
from copy import deepcopy
from functools import partial
from itertools import islice, product, repeat
import multiprocessing
import os
from pprint import pformat
import random
import sys
from threading import Thread
import warnings
import weakref

try:
    from _thread import interrupt_main
except ImportError:
    from thread import interrupt_main

import numpy as np
from scipy import stats
import scipy.linalg
import six
import six.moves as sixm

# helper to cache loading of Stan models
from rstan_interface import get_model, sample
_stan_models = {}
_dir = os.path.abspath(os.path.dirname(__file__))
def get_stan_model(filename='bpmf.stan'):
    try:
        return _stan_models[filename]
    except KeyError:
        _stan_models[filename] = m = get_model(os.path.join(_dir, filename))
        return m


def rmse(a, b):
    diff = a - b
    diff **= 2
    return np.sqrt(np.mean(diff))


def binary_misclassification(a, b):
    return np.mean(np.sign(a) != b)


def project_psd(mat, min_eig=0, destroy=False):
    '''
    Project a real symmetric matrix to PSD by discarding any negative
    eigenvalues from its spectrum. Passing min_eig > 0 lets you similarly make
    it positive-definite, though this may not technically be a projection...?

    Symmetrizes the matrix before projecting.

    If destroy is True, turns the passed-in matrix into gibberish. If the
    matrix is very large, passing in a weakref.proxy to it will use the least
    amount of memory.
    '''
    if not destroy:
        mat = mat.copy()
    mat += mat.T
    mat /= 2

    # TODO: be smart and only get negative eigs?
    vals, vecs = scipy.linalg.eigh(mat)
    if vals.min() < min_eig:
        del mat
        mat = np.dot(vecs, np.dot(np.diag(np.maximum(vals, min_eig)), vecs.T))
        del vals, vecs
        mat += mat.T
        mat /= 2
    return mat


DEFAULT_MLE_EPS = 1e-3
def matrix_normal_mle(samples, eps_u=DEFAULT_MLE_EPS, eps_v=DEFAULT_MLE_EPS,
                      overwrite_samples=False, verbose=False,
                      max_steps=None):
    '''
    Gets maximum likelihood estimates of the parameters of a matrix-normal
    distribution, i.e. the mean, U, and V such that the samples are most likely
    to have come from  N(mean, kronecker(V, U)).

     using the alternating algorithm of
        Pierre Dutilleul. The mle algorithm for the matrix normal distribution.
        Journal of Statistical Computation and Simulation,
        1999 vol. 64 (2) pp. 105-123.
        http://www.tandfonline.com/doi/abs/10.1080/00949659908811970

    The covariance parameters are only unique up to their Kronecker product.

    eps: stopping criterion for the Frobenius norm of the change in the
         covariance factors.

    overwrite_samples: if true, centers the passed samples in-place.
    '''
    r, n, p = samples.shape  # r samples of n x p matrices
    # rp = r * p
    # rn = r * n
    if r <= max(n / p, p / n):
        warnings.warn("Too few samples; MLE doesn't exist, may not converge.")

    mean = np.mean(samples, axis=0)
    if overwrite_samples:
        samples -= mean
    else:
        samples = samples - mean

    old_v = old_u = np.inf

    # starting condition: V = I_p
    v = np.eye(p)

    # U <- \sum x V^-1 x^T
    # u = np.einsum('aij,jk,alk->il', samples, np.linalg.inv(v) / rp, samples)
    u = sum(np.dot(x, x.T) for x in samples)

    frob = partial(np.linalg.norm, ord='fro')
    step = 0

    while frob(v - old_v) > eps_u or frob(u - old_u) > eps_v:
        step += 1
        if verbose >= 2:
            print("\tstep {}: u diff {}, v diff {}".format(
                step, frob(u - old_u), frob(v - old_v)))

        if max_steps and step > max_steps:
            msg = "matrix_normal_mle hit iteration limit ({})".format(max_steps)
            warnings.warn(msg)
            break

        old_u = u
        old_v = v

        # Doing this with a python loop because len(samples) isn't too large,
        # but the size of the matrices could be.
        # http://stackoverflow.com/q/14783386/344821

        try:
            u_cho = scipy.linalg.cho_factor(u)
        except np.linalg.LinAlgError:
            u = project_psd(weakref.proxy(u), min_eig=1e-6, destroy=True)
            u_cho = scipy.linalg.cho_factor(u)
        v = sum(np.dot(x.T, scipy.linalg.cho_solve(u_cho, x)) for x in samples)

        try:
            v_cho = scipy.linalg.cho_factor(v)
        except np.linalg.LinAlgError:
            v = project_psd(weakref.proxy(v), min_eig=1e-6, destroy=True)
            v_cho = scipy.linalg.cho_factor(v)
        u = sum(np.dot(x, scipy.linalg.cho_solve(v_cho, x.T)) for x in samples)

    # No python loop here, but uses an explicit inverse....
    # # V <- \sum x^T U^-1 x
    # v = np.einsum('aji,jk,akl->il', samples, np.linalg.inv(u) / rn, samples)
    # # U <- \sum x V^-1 x^T
    # u = np.einsum('aij,jk,alk->il', samples, np.linalg.inv(v) / rp, samples)

    if verbose:
        print("\tmatrix_normal_mle took {} steps".format(step))

    return mean, u, v

################################################################################

class BPMF(object):
    def __init__(self, rating_tuples, latent_d,
                 subtract_mean=True,
                 rating_values=None,
                 discrete_expectations=True,
                 num_integration_pts=50,
                 knowable=None,
                 model_filename='bpmf_w0identity.stan'):
        self.latent_d = latent_d
        self.subtract_mean = subtract_mean

        self.rating_std = 1/2
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
        self.model_filename = model_filename or 'bpmf_w0identity.stan'

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
        # this could be done without a copy by .resize()...

        # keep the old sampled mode, but immediately replace it once we have
        # a new one, since the old log-likelihood is no longer valid
        self.sampled_mode_lp = -np.inf

    def _data_for_sampling(self):
        ratings = self.ratings[:, 2]
        if self.subtract_mean:
            ratings = ratings - self.mean_rating
        return {
            'n_users': self.num_users,
            'n_items': self.num_items,
            'rank': self.latent_d,

            'n_obs': self.ratings.shape[0],
            'obs_users': self.ratings[:, 0] + 1,
            'obs_items': self.ratings[:, 1] + 1,
            'obs_ratings': ratings,

            'rating_std': self.rating_std,
            'mu_0': self.mu_0,
            'beta_0': self.beta_0,
            'nu_0': self.nu_0,
            'w_0': self.w_0,
        }

    def _fill_predictions(self, samps):
        samps['predictions'] = np.einsum('aij,akj->aik', samps['U'], samps['V'])

    def samples(self, num_samps, warmup=None, chains=1,
                start_at_mode=True, update_mode=True, model_filename=None,
                eat_output=True,
                ret_args_only=False):
        '''
        Runs the Markov chain for num_samps samples, after warming up for
        warmup iterations beforehand (default: num_samps // 2).

        Returns a dictionary with keys the name of the param, values
        an array with first dimension num_samps.
        '''
        # TODO: options about starting at previous points...
        if warmup is None:
            warmup = num_samps // 2

        data = self._data_for_sampling()
        args = {'chains': chains, 'iter': warmup + num_samps, 'warmup': warmup,
                'eat_output': eat_output, 'return_output': False}
        if start_at_mode:
            args['init'] = self.sampled_mode

        if ret_args_only:
            return data, args

        stan_model = get_stan_model(model_filename or self.model_filename)
        samples = sample(stan_model, data=data, **args)
        # TODO: would be nice to initialize step size parameters to old values
        #       but still allow adaptation. unfortunately, stan doesn't
        #       currently support this.

        if 'predictions' not in samples:
            self._fill_predictions(samples)

        if update_mode:
            i = samples['lp__'].argmax()
            if samples['lp__'][i] > self.sampled_mode_lp:
                self.sampled_mode = dict(
                    (k, v[i]) for k, v in six.iteritems(samples))
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
        if not hasattr(which, 'shape') and which == Ellipsis:
            preds = samples['predictions']
        else:
            preds = np.asarray([p[which] for p in samples['predictions']])
        return (preds + self.mean_rating) if self.subtract_mean else preds

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

    def entropy_est(self, samples, which=Ellipsis, eps=DEFAULT_MLE_EPS,
                    additive_constant=False):
        '''
        Estimates the entropy of the samples by assuming that the prediction
        matrix is distributed according to a matrix-normal distribution.

        Doesn't bother with the additive constant by default.

        NOTE XXX: *ignores* the which argument!
        '''
        # TODO: can we center the samples in-place here?
        _, u, v = matrix_normal_mle(self.pick_out_predictions(samples),
                                    eps_u=eps, eps_v=eps,
                                    max_steps=1000)

        sign_det_u, logdet_u = np.linalg.slogdet(u)
        sign_det_v, logdet_v = np.linalg.slogdet(v)
        entropy = self.num_items * logdet_u + self.num_users * logdet_v

        if additive_constant:
            entropy += (1 + np.log(2 * np.pi)) * self.num_items * self.num_users
        return entropy / 2

    def exp_variance(self, samples, which=Ellipsis, pool=None,
                     num_samps=30, warmup=15, **sample_args):
        '''
        Gives the total expected variance in our predicted R matrix if we knew
        Rij for each ij in "which", using samples to represent our distribution
        for Rij: \sum E[Var[R_ij]].

        Parallelizes the evaluation over pool if passed (highly recommended,
        since this is sloooow).
        '''
        return self._distribute(_exp_variance_helper,
                samples=samples, which=which, pool=pool,
                num_samps=num_samps, warmup=warmup, **sample_args)

    def exp_entropy_est(self, samples, which=Ellipsis, pool=None,
                        num_samps=30, warmup=15, **sample_args):
        '''
        Gives the expected entropy in our predicted R matrix if we knew
        Rij for each ij in "which", using samples to represent our distribution
        for Rij: \sum E[Var[R_ij]].

        Parallelizes the evaluation over pool if passed (highly recommended,
        since this is sloooow).
        '''
        return self._distribute(_exp_entropy_est_helper,
                samples=samples, which=which, pool=pool,
                num_samps=num_samps, warmup=warmup, **sample_args)

    def _distribute(self, fn, samples, which, pool, **sample_args):
        # figure out the indices of our selection
        # ...there's gotta be an easier way, right?
        n = self.num_users
        m = self.num_items
        i_indices = np.repeat(np.arange(n).reshape(n, 1), m, axis=1)[which]
        j_indices = np.repeat(np.arange(m).reshape(1, m), n, axis=0)[which]

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
        evals = np.array([calculate_fn(v) if p else 0
                          for p, v in sixm.zip(params, bpmf.rating_values)])
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
    try:
        return _integrate_lookahead(BPMF.total_variance, *args, **sample_args)
    except:
        import traceback
        traceback.print_exc()
        raise

def _exp_entropy_est_helper(args, **sample_args):
    try:
        return _integrate_lookahead(BPMF.entropy_est, *args, **sample_args)
    except:
        import traceback
        traceback.print_exc()
        raise


################################################################################

# The various evaluation functions we consider
Key = namedtuple('Key', 'nice_name key_fn choose_max does_sampling args')
KEYS = {
    'random': Key("Random", 'random', True, False, ()),
    'pred-variance': Key("Var[R_ij]", 'pred_variance', True, False, ()),

    'exp-variance': Key("E[Var[R]]", 'exp_variance', False, True, ()),
    'exp-entropy-est': Key("E[H[R]]", 'exp_entropy_est', False, True, ()),

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
              pool=None, sample_in_pool=False, test_on=Ellipsis,
              binary_acc=False):
    '''
    Evaluates the chosen selection criterion (key_name) on the data (real),
    starting with an initial instance of BPMF and some samples from it.

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

    real_test = real[test_on]

    init_pred = bpmf.predict(samples)
    if binary_acc:
        assert np.all(np.abs(real[test_on])) == 1
        init_err = binary_misclassification(init_pred[test_on], real_test)
    else:
        init_err = rmse(init_pred[test_on], real_test)
    yield (len(bpmf.rated), init_err, None, None, init_pred)

    status = partial(print, "{:<40}".format(key.nice_name))

    while bpmf.unrated:
        status("Picking query point {}...".format(len(bpmf.rated) + 1))

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
        status("Queried ({}, {}); {}/{} known".format(
                i, j, len(bpmf.rated), total))

        if sample_in_pool and pool:
            samples, pred = pool.apply(fetch_samples, [bpmf], samp_args)
        else:
            samples, pred = fetch_samples(bpmf, **samp_args)

        if binary_acc:
            err = binary_misclassification(pred[test_on], real_test)
            status("Error rate {}: {:.3%}".format(len(bpmf.rated), err))
        else:
            err = rmse(pred[test_on], real[test_on])
            status("RMSE {}: {:.5}".format(len(bpmf.rated), err))
        yield len(bpmf.rated), err, (i, j), vals, pred


class MainProgram(object):
    def get_parser(self):
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

        # set up arguments
        parser = argparse.ArgumentParser()

        parser.add_argument('--latent-d', '-D', type=int, default=5,
            help="The rank of the factorization. Default %(default)s.")
        parser.add_argument('--steps', '-s', type=int, default=None,
            help="How many active learning steps to perform. "
                 "Default: until fully known.")

        parser._add_action(ActionNoYes('discrete', default=None,
            help="Whether the output matrix is discrete (affects the "
                 "integration during lookahead). Default: true if the data "
                 "file contains _rating_vals."))
        parser.add_argument('--num-integration-pts', type=int, default=50,
            help="Number of points to do in numerical integration for "
                 "continuous outputs; default %(default)s.")

        parser._add_action(ActionNoYes('binary-acc', default=False,
            help="Evaluate accuracy by treating positive outputs as 1 and "
                 "negative outputs as 0, and getting the accuracy. Default: "
                 "evaluate accuracy via RMSE."))

        parser._add_action(ActionNoYes('subtract-mean', default=True,
            help="Center the output matrix; default yes. Usually a good idea, "
                 "but can cause problems when there are very few known "
                 "points."))

        parser.add_argument('--hyperparams', default={}, type=lambda x: eval(x),
            help="Hyperparameters (class attributes of the BPMF object), "
                 "written as a literal python dictionary.")
        parser._add_action(ActionNoYes('initialize-at-pmf-map', default=False,
            help="Start the learning process at a MAP estimate, obtained via "
                 "the code in ../python-pmf/pmf.py. Default false."))

        parser.add_argument('--samps', '-S', type=int, default=100,
            help="How many samples to take for MCMC inference; "
                 "default %(default)s.")
        parser.add_argument('--warmup', type=int, default=50,
            help="How many samples to warm up with for MCMC. More important "
                 "for NUTS than traditional MCMC because it adapts parameters "
                 "only during warmup. Default %(default)s.")

        parser.add_argument('--lookahead-samps', type=int, default=100,
            help="How many samples to take during lookahead MCMC.")
        parser.add_argument('--lookahead-warmup', type=int, default=50,
            help="How much warmup to do during lookahead MCMC.")

        parser._add_action(ActionNoYes('threaded', 'unthreaded', default=True,
            help="Evaluate different criteria in different threads, rather "
                 "than sequentially. Default: true."))
        parser.add_argument('--procs', '-P', type=int, default=None,
            help="How many total processes to fork for evaluation. These are "
                 "shared across all the threads if --threaded. Default: one "
                 "per CPU core.")

        parser._add_action(ActionNoYes('test-set-from-file', default=True,
            help="Load test sets from the data file."))
        parser.add_argument('--test-set', default="all",
            help="'all' (the default, meaning get test evals on all known "
                 "points), "
                 "an integer (meaning choose that many points to make a "
                 "a distinct test set), "
                 "or a real 0 < x <= 1 (to choose that portion of points for a "
                 "distinct test set. "
                 "If --test-set-from-file this is overridden by any "
                 "in-file test sets.")

        parser._add_action(ActionNoYes('query-new-only', default=False,
            help="Only query points that are labeled 'new' in _is_new_item "
                 "in the data file. Default off."))

        parser.add_argument('--model-filename', default=None,
            help="The filename to use as a Stan model for BPMF. Default: "
                 "bpmf_w0identity.stan.")

        parser.add_argument('--load-data', required='True', metavar='FILE',
            help="Load data from this file. Should be either an npz archive "
                 "or a pickle file of a dictionary, containing keys "
                 "_real (an output data matrix, where 0 or nan means unknown), "
                 "_ratings (an n x 3 array of [row, col, value] initially "
                    "known entries), "
                 "_rating_vals (optional, the list of possible output vals), "
                 "_test_on (optional, a boolean mask array of same shape as "
                    "_real defining which points are the test set), "
                 "and _is_new_item (optional, a vector determining whether "
                    "each column is 'new').")
        parser.add_argument('--save-results', nargs='?', default=True,
                const=True, metavar='FILE',
                help="Save results to FILE, or results.pkl if no FILE given.")
        parser.add_argument('--no-save-results',
                action='store_false', dest='save_results',
                help="Don't save any results.")

        parser.add_argument('--note', action='append',
            help="Doesn't do anything, just there to save any notes you'd like "
                 "in the results file.")
        parser._add_action(ActionNoYes('pdb-on-error', default=True,
            help="If there's an error somewhere, drop down into the debugger."))

        parser.add_argument('keys', nargs='*',
                help="The selection criteria to try. "
                     "Choices: {}.".format(', '.join(sorted(KEYS.keys()))))
        return parser

    def parse_args(self):
        import os
        parser = self.get_parser()
        args = parser.parse_args()

        # check that args.keys are valid
        key_names = KEYS.keys()
        for k in args.keys:
            if k not in key_names:
                parser.error("Invalid key name %s; options are %s.\n" % (
                    k, ', '.join(sorted(key_names))))

        if not args.keys:
            args.keys = sorted(key_names)

        # make directories to save results if necessary
        if args.save_results is True:
            args.save_results = 'results.pkl'
        elif args.save_results:
            dirname = os.path.dirname(args.save_results)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

        return args

    def load_data(self, args):
        # load data
        with open(args.load_data, 'rb') as f:
            data = np.load(f)
            get = lambda k: data[k] if k in data else None

            real = data['_real']
            ratings = data['_ratings']
            rating_vals = get('_rating_vals')
            test_on = get('_test_on')
            is_new_item = get('_is_new_item')

        if args.test_set_from_file and (test_on is not None):
            test_set = test_on
        else:
            try:
                test_set = int(args.test_set)
            except ValueError:
                try:
                    test_set = float(args.test_set)
                except ValueError:
                    test_set = args.test_set

        if args.discrete is None:
            args.discrete = rating_vals is not None

        Data = namedtuple("Data", "real ratings rating_vals test_set is_new_item")
        return Data(real, ratings, rating_vals, test_set, is_new_item)

    def initialize_bpmf(self, args, data, query_set):
        bpmf = BPMF(data.ratings, args.latent_d,
                subtract_mean=args.subtract_mean,
                rating_values=data.rating_vals,
                discrete_expectations=args.discrete,
                num_integration_pts=args.num_integration_pts,
                knowable=query_set,
                model_filename=args.model_filename)
        for k, v in args.hyperparams.items():
            assert hasattr(bpmf, k)
            setattr(bpmf, k, v)

        if args.initialize_at_pmf_map:
            import sys
            sys.path.append(
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../python-pmf')))
            try:
                from pmf_cy import ProbabilisticMatrixFactorization
            except ImportError:
                warnings.warn("cython PMF not available; "
                              "using pure-python version")
                from pmf import ProbabilisticMatrixFactorization

            print("getting PMF MAP fit with default hyperparams")
            p = ProbabilisticMatrixFactorization(
                data.ratings, latent_d=args.latent_d,
                subtract_mean=args.subtract_mean)
            p.fit()  # TODO: support also doing SGD

            print("okay; BPMF will start from here now")
            # set the mode to the PMF mode
            bpmf.samples(num_samps=1, warmup=0, update_mode=True)
            bpmf.sampled_mode['U'] = p.users
            bpmf.sampled_mode['V'] = p.items
            bpmf.sampled_mode['predictions'] = p.predicted_matrix()
            bpmf.sampled_mode_lp = -np.inf

        return bpmf

    def pick_query_test_sets(self, args, data):
        real = data.real
        ratings = data.ratings
        test_set = data.test_set
        rating_vals = data.rating_vals

        # figure out which points we know the answers to
        knowable = np.isfinite(real)
        knowable[real == 0] = 0

        # ... and which points we can query
        pickable = knowable.copy()
        pickable[ratings[:, 0].astype(int), ratings[:, 1].astype(int)] = 0

        # TODO: support specifying querying on a subset of available things,
        #       other than just complement of the test_set
        if test_set == 'all':
            test_on = knowable
            query_on = pickable

        elif np.isscalar(test_set) and np.asarray(test_set).dtype.kind in "fiu":
            if 0 < test_set <= 1:
                test_set = int(np.round(test_set * pickable.size))
            elif test_set == np.round(test_set):
                test_set = int(test_set)
            else:
                msg = "can't interpret test_set {!r}".format(test_set)
                raise TypeError(msg)

            avail_pts = list(sixm.zip(*pickable.nonzero()))
            picked_indices = random.sample(avail_pts, test_set)
            picker = np.zeros(pickable.shape, bool)
            picker[tuple(np.transpose(picked_indices))] = 1

            test_on = picker * pickable
            query_on = (1 - picker) * pickable

        else:
            if hasattr(test_set, 'shape') and test_set.shape == knowable.shape:
                picker = test_set.astype(bool)
            else:
                picker = np.zeros(knowable.shape, dtype=bool)
                try:
                    picker[test_set] = True
                except IndexError:
                    msg = "can't interpret test_set {!r}".format(test_set)
                    raise TypeError(msg)
            test_on = picker * knowable
            query_on = ~picker * pickable

        if args.query_new_only:
            assert data.is_new_item is not None
            query_on[:, ~data.is_new_item] = False

        print("{} users, {} items".format(*real.shape))
        print("{} points known, querying {}, testing {}, {} knowable, {} total"
                .format(ratings.shape[0], query_on.sum(), test_on.sum(),
                        knowable.sum(), real.size))

        test_query = np.sum(test_on & query_on)
        if test_query:
            print("test, query set have {} common elements".format(test_query))
        else:
            print("test and query sets are distinct")

        if rating_vals is not None:
            known = np.zeros_like(test_on, dtype=bool)
            known[ratings[:, 0].astype(int), ratings[:, 1].astype(int)] = True

            for s, thing in [("known", known),
                             ("test", test_on),
                             ("query", query_on)]:
                counts = Counter(real[thing].flat)
                counts.update(dict((k, 0) for k in rating_vals))
                print("{} set distribution: {}"
                        .format(s, pformat(dict(counts))))

        return query_on, test_on

    def do_work(self, args, data):
        real = data.real
        ratings = data.ratings
        rating_vals = data.rating_vals
        query_on, test_on = self.pick_query_test_sets(args, data)
        query_set = set(sixm.zip(*query_on.nonzero()))

        bpmf_init = self.initialize_bpmf(args, data, query_set)

        if not args.procs:
            pool = multiprocessing.Pool()
        elif args.procs == 1:
            pool = None
        else:
            pool = multiprocessing.Pool(args.procs)

        samp_args = {'warmup': args.warmup}

        print("Getting initial MCMC samples...")
        samples = bpmf_init.samples(num_samps=args.samps, **samp_args)

        init_pred_on_test = bpmf_init.predict(samples, which=test_on)
        if args.binary_acc:
            assert np.all(np.abs(real[test_on])) == 1
            init_err = binary_misclassification(init_pred_on_test, real[test_on])
            print("Initial error rate: {:.3%}".format(init_err))
        else:
            init_err = rmse(init_pred_on_test, real[test_on])
            print("Initial RMSE: {}".format(init_err))
        print()

        results = {
            '_real': real,
            '_ratings': ratings,
            '_rating_vals': rating_vals,
            '_initial_bpmf': deepcopy(bpmf_init),
            '_test_on': test_on,
            '_query_on': query_on,
        }

        # continue with each key for the fit
        def eval_key(key_name):
            try:
                res = full_test(
                    deepcopy(bpmf_init), samples, real, key_name,
                    pool=pool, sample_in_pool=args.threaded,
                    test_on=test_on, binary_acc=args.binary_acc,
                    num_samps=args.samps, samp_args=samp_args,
                    lookahead_samps=args.lookahead_samps,
                    lookahead_samp_args={'warmup': args.lookahead_warmup})
                results[key_name] = list(islice(res, args.steps))
                # TODO: save results / some state along the way
            except BaseException:
                import traceback
                traceback.print_exc()
                if pool is not None:
                    pool.terminate()
                if args.threaded:
                    interrupt_main()
                raise

        # TODO: no concurrent access to R. be careful here...
        if args.threaded:
            threads = [Thread(name=key_name, target=eval_key, args=(key_name,))
                       for key_name in args.keys]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        else:
            for key_name in args.keys:
                eval_key(key_name)

        if pool is not None:
            pool.close()
            pool.join()

        return results

    def main(self):
        import pickle

        try:
            args = self.parse_args()
            the_data = self.load_data(args)
            results = self.do_work(args, the_data)
        except Exception:
            if not args.pdb_on_error:
                raise

            import traceback
            traceback.print_exc()

            try:
                import ipdb as pdb
            except ImportError:
                import pdb
            print()
            pdb.post_mortem(sys.exc_info()[2])

            sys.exit(1)

        # save the results file
        if args.save_results:
            print("\nsaving results in '{}'".format(args.save_results))

            results['_args'] = args

            with open(args.save_results, 'wb') as f:
                pickle.dump(results, f, protocol=2)

if __name__ == '__main__':
    MainProgram().main()
