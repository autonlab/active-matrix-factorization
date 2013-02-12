#!/usr/bin/env python3
'''
Implementation of Bayesian PMF (via Gibbs sampling).

Based on Matlab code by Ruslan Salakhutdinov:
http://www.mit.edu/~rsalakhu/BPMF.html

NOTE: this should be considered depracated;
      the HMC-based sampler runs much faster and seems to sample better
'''

from __future__ import print_function # silly cython

from collections import defaultdict, namedtuple
from copy import deepcopy
from itertools import islice, repeat
import multiprocessing
import random
from threading import Thread
import warnings

import numpy as np
from scipy import stats, integrate

# TODO: make this actually work....
#if not cython.compiled:
#    try:
#        from pmf_cy import ProbabilisticMatrixFactorization
#    except ImportError:
#        warnings.warn("cython PMF not available; using pure-python version")
#        from pmf import ProbabilisticMatrixFactorization
from pmf_cy import ProbabilisticMatrixFactorization, rmse, parse_fit_type

import cython

################################################################################
### Utilities

# This function by Matthew James Johnson, from:
# http://www.mit.edu/~mattjj/released-code/hsmm/stats_util.py
def sample_wishart(sigma, dof):
    '''
    Returns a sample from the Wishart distribution, the conjugate prior for
    precision matrices.
    '''
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling
    # schemes
    if dof <= 81+n and dof == round(dof):
        # direct
        X = np.dot(chol, np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X, X.T)


def iter_mean(iterable):
    i = iter(iterable)
    total = next(i)
    count = -1
    for count, x in enumerate(i):
        total += x
    return total / (count + 2)

################################################################################

class BayesianPMF(ProbabilisticMatrixFactorization):
    def __init__(self, rating_tuples, latent_d=5,
                 subtract_mean=True,
                 rating_values=None,
                 discrete_expectations=True,
                 num_integration_pts=50,
                 knowable=None,
                 fit_type=('batch',)):

        super(BayesianPMF, self).__init__(
                rating_tuples, latent_d=latent_d,
                subtract_mean=subtract_mean,
                knowable=knowable, fit_type=fit_type)

        if rating_values is not None:
            rating_values = set(map(float, rating_values))
            if not rating_values.issuperset(self.ratings[:,2]):
                raise ValueError("got ratings not in rating_values")
        self.rating_values = rating_values
        self.discrete_expectations = discrete_expectations
        self.num_integration_pts = num_integration_pts

        self.beta = 2 # observation noise precision

        # parameters of inverse-wishart
        self.u_hyperparams = (
            np.eye(latent_d), # wi = wishart scale matrix (latent_d x latent_d)
            2, # b0 = scale on the gaussian's precision (scalar)
            latent_d, # degrees of freedom
            np.zeros(latent_d), # mu0 = mean of gaussian
        )

        self.v_hyperparams = (
            np.eye(latent_d), # wi = wishart scale matrix (latent_d x latent_d)
            2, # b0 = scale on the gaussian's precision (scalar)
            latent_d, # degrees of freedom
            np.zeros(latent_d), # mu0 = mean of gaussian
        )

    def __copy__(self):
        # need to copy fields from super
        res = BayesianPMF(self.ratings, self.latent_d)
        res.__setstate__(self.__getstate__())
        return res

    def __deepcopy__(self, memodict):
        # need to copy fields from super
        res = BayesianPMF(self.ratings, self.latent_d)
        res.__setstate__(deepcopy(self.__getstate__(), memodict))
        return res

    def __getstate__(self):
        state = super(BayesianPMF, self).__getstate__()
        if cython.compiled:
            state['discrete_expectations'] = self.discrete_expectations
            state['rating_values'] = self._rating_values
            state['beta'] = self.beta
            state['u_hyperparams'] = self.u_hyperparams
            state['v_hyperparams'] = self.v_hyperparams
        else:
            #state.update(self.__dict__)
            state['__dict__'] = self.__dict__
        return state


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


    def sample_hyperparam(self, feats, do_users):
        '''
        Samples a mean hyperparameter conditional on the feature matrix
        (Gaussian-Wishart distribution).

        User hyperparams if do_users, otherwise item hyperparams.
        '''

        wi, b0, df, mu0 = self.u_hyperparams if do_users else self.v_hyperparams

        N = feats.shape[0]
        x_bar = np.mean(feats, axis=0).T
        S_bar = np.cov(feats, rowvar=0)

        mu0_xbar = mu0 - x_bar

        WI_post = np.linalg.inv(
                np.linalg.inv(wi)
                + N * S_bar
                + (b0 * N) / (b0 + N) * np.dot(mu0_xbar, mu0_xbar.T))
        WI_post /= 2
        WI_post = WI_post + WI_post.T

        alpha = sample_wishart(WI_post, df + N)

        mu_temp = (b0 * mu0 + N * x_bar) / (b0 + N)
        lam = np.linalg.cholesky(np.linalg.inv((b0 + N) * alpha))
        mu = np.dot(lam, np.random.normal(0, 1, self.latent_d)) + mu_temp

        return mu, alpha


    def sample_feature(self, n, is_user, mu, alpha, oth_feats,
                       rated_indices, ratings):
        '''
        Samples a user/item feature vector, conditional on the entire
        matrix of other item/user features.

        n: the id of the user/item
        is_user: true if this is a user, false if an item
        mu: the mean hyperparameter for users if is_user, items if not
        alpha: the precision hyperparamater
        oth_feats: self.items/self.users
        rated_indices: indices of the items rated by this user / users
                       who rated this item
        ratings: ratings by this user / for this item for rated_indices
        '''

        rated_feats = oth_feats[rated_indices, :]
        if self.subtract_mean:
            ratings = ratings - self.mean_rating

        cov = np.linalg.inv(alpha +
                self.beta * np.dot(rated_feats.T, rated_feats))
        mean = np.dot(cov,
                self.beta * np.dot(rated_feats.T, ratings)
                + np.dot(alpha, mu))

        lam = np.linalg.cholesky(cov)
        return np.dot(lam, np.random.normal(0, 1, self.latent_d)) + mean

    @cython.locals(
        num_gibbs=cython.int,
        users_by_item=dict, items_by_user=dict,
        user_sample=np.ndarray, item_sample=np.ndarray,
        mu_u=np.ndarray, mu_v=np.ndarray,
        alpha_u=np.ndarray, alpha_v=np.ndarray,
        rated_indices=np.ndarray, ratings=np.ndarray,
        user_id=cython.int, item_id=cython.int,
    )
    def samples(self, num_gibbs=2, fit_first=False):
        '''
        Runs the Markov chain starting from the current MAP approximation in
        self.users, self.items. Yields sampled user, item features forever.

        If you add ratings after starting this iterator, it'll continue on
        without accounting for them.

        Does num_gibbs updates after each hyperparameter update, then yields
        the result.

        If fit_first is True, calls self.do_fit() to fit the MAP estimate.
        '''
        # find rated indices now, to avoid repeated lookups
        users_by_i = defaultdict(lambda: ([], []))
        items_by_u = defaultdict(lambda: ([], []))

        for user, item, rating in self.ratings:
            users_by_i[item][0].append(user)
            users_by_i[item][1].append(rating)

            items_by_u[user][0].append(item)
            items_by_u[user][1].append(rating)

        users_by_item = {k: (np.asarray(i, dtype=int), np.asarray(r))
                         for k, (i,r) in users_by_i.items()}
        items_by_user = {k: (np.asarray(i, dtype=int), np.asarray(r))
                         for k, (i,r) in items_by_u.items()}
        del users_by_i, items_by_u

        # fit the MAP estimate, if asked to
        if fit_first:
            self.do_fit()

        # initialize the Markov chain with the current MAP estimate
        user_sample = self.users.copy()
        item_sample = self.items.copy()

        # mu is the average value for each latent dimension
        mu_u = np.mean(user_sample, axis=0).T
        mu_v = np.mean(item_sample, axis=0).T

        # alpha is the inverse covariance among latent dimensions
        if self.latent_d == 1:
            alpha_u = np.array([1 / np.var(user_sample, ddof=1)])
            alpha_v = np.array([1 / np.var(item_sample, ddof=1)])
        else:
            alpha_u = np.linalg.inv(np.cov(user_sample, rowvar=0))
            alpha_v = np.linalg.inv(np.cov(item_sample, rowvar=0))

        while True:
            # sample from hyperparameters
            mu_u, alpha_u = self.sample_hyperparam(user_sample, True)
            mu_v, alpha_v = self.sample_hyperparam(item_sample, False)

            # Gibbs updates for user, item feature vectors
            for gibbs in range(num_gibbs):

                user_sample = np.empty_like(user_sample)
                for user_id in range(self.num_users):
                    rated_indices, ratings = items_by_user[user_id]

                    user_sample[user_id] = self.sample_feature(
                            user_id, True, mu_u, alpha_u, item_sample,
                            rated_indices, ratings
                    )

                item_sample = np.empty_like(item_sample)
                for item_id in range(self.num_items):
                    rated_indices, ratings = users_by_item[item_id]

                    item_sample[item_id] = self.sample_feature(
                            item_id, False, mu_v, alpha_v, user_sample,
                            rated_indices, ratings)

            yield user_sample, item_sample



    def samples_parallel(self, num_gibbs=2, pool=None, multiproc_mode=None,
                fit_first=False):
        '''
        Runs the Markov chain starting from the current MAP approximation in
        self.users, self.items. Yields sampled user, item features forever.

        If you add ratings after starting this iterator, it'll continue on
        without accounting for them.

        Does num_gibbs updates after each hyperparameter update, then yields
        the result.

        Optionally parallelizes sampling according to multiproc_mode:

          * If 'force', all significant work will be executed in the pool,
            even things that aren't particularly parallelizable. (This is
            so the multithreading/multiprocessing setup works.) Throws a
            ValueError if no pool is passed.

          * If 'none', no multiprocessing is performed and the value of pool
            is ignored.

          * If None (default), will perform user/item parallelization in the
            pool if passed, or no parallelization if the pool is not passed.

        If fit_first is True, first calls .do_fit() to fit the MAP estimate.
        If multiproc_mode is 'force', offloads this fitting to the pool.
        '''

        if multiproc_mode == 'force' and pool is None:
            raise ValueError("need a process pool if multiproc is forced")
        if multiproc_mode == 'none':
            pool = None
        force_multiproc = multiproc_mode == 'force'

        # find rated indices now, to avoid repeated lookups
        users_by_item = defaultdict(lambda: ([], []))
        items_by_user = defaultdict(lambda: ([], []))

        for user, item, rating in self.ratings:
            users_by_item[item][0].append(user)
            users_by_item[item][1].append(rating)

            items_by_user[user][0].append(item)
            items_by_user[user][1].append(rating)

        users_by_item = {k: (np.asarray(i, dtype=int), np.asarray(r))
                         for k, (i,r) in users_by_item.items()}
        items_by_user = {k: (np.asarray(i, dtype=int), np.asarray(r))
                         for k, (i,r) in items_by_user.items()}

        # fit the MAP estimate, if asked to
        if fit_first:
            if force_multiproc:
                bpmf = pool.apply(_fit_pmf)
                self.users = bpmf.users
                self.items = bpmf.items
            else:
                self.do_fit()

        # initialize the Markov chain with the current MAP estimate
        user_sample = self.users.copy()
        item_sample = self.items.copy()

        # mu is the average value for each latent dimension
        mu_u = np.mean(user_sample, axis=0).T
        mu_v = np.mean(item_sample, axis=0).T

        # alpha is the inverse covariance among latent dimensions
        if self.latent_d == 1:
            alpha_u = np.array([1 / np.var(user_sample, ddof=1)])
            alpha_v = np.array([1 / np.var(item_sample, ddof=1)])
        else:
            alpha_u = np.linalg.inv(np.cov(user_sample, rowvar=0))
            alpha_v = np.linalg.inv(np.cov(item_sample, rowvar=0))

        # TODO: could try using pool.imap if memory becomes an issue
        # could also use map_async
        # or manually distribute chunks of rows to speed up computation
        mapper = pool.map if pool is not None else map

        while True:
            # sample from hyperparameters
            if force_multiproc:
                r1 = pool.apply_async(_hyperparam_sampler,
                        (self, user_sample, True))
                r2 = pool.apply_async(_hyperparam_sampler,
                        (self, item_sample, False))

                mu_u, alpha_u = r1.get()
                mu_v, alpha_v = r2.get()
            else:
                mu_u, alpha_u = self.sample_hyperparam(user_sample, True)
                mu_v, alpha_v = self.sample_hyperparam(item_sample, False)

            # Gibbs updates for user, item feature vectors
            for gibbs in range(num_gibbs):
                #print('\t\t Gibbs sampling {}'.format(gibbs))

                res = mapper(_feat_sampler,
                        ((self, user_id, True, mu_u, alpha_u, item_sample)
                            + items_by_user[user_id]
                         for user_id in range(self.num_users)))

                user_sample = np.zeros_like(user_sample)
                for n, row in enumerate(res):
                    user_sample[n, :] = row


                res = mapper(_feat_sampler,
                        ((self, item_id, False, mu_v, alpha_v, user_sample)
                            + users_by_item[item_id]
                         for item_id in range(self.num_items)))

                item_sample = np.zeros_like(item_sample)
                for n, row in enumerate(res):
                    item_sample[n, :] = row

            yield user_sample, item_sample


    def matrix_results(self, vals, which):
        res = np.empty((self.num_users, self.num_items))
        res.fill(np.nan)
        res[which] = vals
        return res

    def predict(self, samples_iter, which=Ellipsis):
        '''
        Gives the mean reconstruction given a series of samples.
        '''
        return iter_mean(self.predicted_matrix(u, v)[which]
                         for u, v in samples_iter)

    def pred_variance(self, samples_iter, which=Ellipsis):
        '''
        Gives the variance of each prediction in a series of samples.
        '''
        if which is None:
            which = Ellipsis

        vals = [self.predicted_matrix(u, v)[which] for u, v in samples_iter]
        return np.var(vals, 0)

    def total_variance(self, samples_iter, which=Ellipsis):
        '''
        Gives the sum of the variance of each element of the prediction matrix
        selected by which over samples_iter.
        '''
        return self.pred_variance(samples_iter, which=which).sum()

    def exp_variance(self, samples_iter, which=Ellipsis, pool=None,
                     fit_first=True, num_samps=30):
        '''
        Gives the total expected variance in our predicted R matrix if we knew
        Rij for each ij in which, using samples_iter to get our distribution
        for Rij.

        Parallelizes the evaluation over pool if passed (highly recommended,
        since this is sloooow).
        '''
        return self._distribute(_exp_variance_helper,
                samples_iter, which, pool, fit_first, num_samps)

    def _distribute(self, fn, samples_iter, which, pool, fit_first, num_samps):
        # figure out the indices of our selection
        # ...there's gotta be an easier way, right?
        n = self.num_users
        m = self.num_items
        all_indices = np.empty((n, m, 2), dtype=int)
        for i in range(n):
            all_indices[i, :, 0] = i
        for j in range(m):
            all_indices[:, j, 1] = j
        indices = all_indices[which]
        i_indices = indices[...,0]
        j_indices = indices[...,1]

        # get samples of R_ij for each ij in which
        vals = np.asarray([
            self.predicted_matrix(u, v)[which] for u, v in samples_iter])

        # fit a distribution to the samples for each ij
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
            params = zip(mean.flat, var.flat)

        # TODO could experiment with map_async, imap_unordered
        mapper = pool.map if pool is not None else map
        exps = mapper(fn,
                zip(repeat(self),
                    i_indices.flat, j_indices.flat,
                    repeat(discrete), params,
                    repeat(fit_first), repeat(num_samps)))

        res = np.empty(np.shape(vals)[1:])
        res.fill(np.nan)
        for idx, exp in enumerate(exps):
            res.flat[idx] = exp
        return res


    def prob_ge_cutoff(self, samples_iter, cutoff, which=Ellipsis):
        '''
        Gives the portion of the time each matrix element was >= cutoff
        in a series of samples.
        '''
        counts = np.zeros((self.num_users, self.num_items), dtype=int)[which]
        num = 0
        for u, v in samples_iter:
            counts += (self.predicted_matrix(u, v)[which] >= cutoff)
            num += 1
        return counts / float(num)

    def random(self, samples_iter, which=Ellipsis):
        shape = np.empty((self.num_users, self.num_items))[which].shape
        return np.random.rand(*shape)

    def bayes_rmse(self, samples_iter, true_r, which=Ellipsis):
        return rmse(self.predict(samples_iter, which), true_r[which])


# stupid functions to work around multiprocessing.Pool/pickle silliness
def _fit_pmf(pmf):
    pmf.do_fit()
    return pmf

def _hyperparam_sampler(bpmf, *args):
    return bpmf.sample_hyperparam(*args)

def _feat_sampler(args):
    bpmf, *args = args
    return bpmf.sample_feature(*args)

def _integrate_lookahead(fn, bpmf, i, j, discrete, params, fit_first, num_samps):
    if (i, j) in bpmf.rated:
        warnings.warn("Asked to check a known entry; returning NaN")
        return np.nan

    def calculate_fn(v):
        b = deepcopy(bpmf)
        b.add_rating(i, j, v)
        samps = b.samples(fit_first=fit_first)
        return fn(b, islice(samps, num_samps))

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
        evals = np.fromiter(map(calculate_fn, pts), float, pts.size)
        est = integrate.trapz(evals * dist.pdf(pts), pts)
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


def _exp_variance_helper(args):
    return _integrate_lookahead(BayesianPMF.total_variance, *args)


################################################################################

def test_vs_map():
    from pmf import fake_ratings

    ratings, true_u, true_v = fake_ratings(noise=1)
    true_r = np.dot(true_u, true_v.T)

    ds = [3, 5, 8, 10, 12, 15]
    map_rmses = []
    bayes_rmses_1 = []
    bayes_rmses_2 = []
    bayes_rmses_combo = []

    for latent_d in ds:
        bpmf = BayesianPMF(ratings, latent_d)

        print("\ndimensionality: {}".format(latent_d))

        print("fitting MAP...")
        for ll in bpmf.fit_lls():
            pass
            #print("LL {}".format(ll))

        predicted_map = bpmf.predicted_matrix()

        print("doing MCMC...")
        samps = list(islice(bpmf.samples(), 500))

        bayes_rmses_1.append(bpmf.bayes_rmse(islice(samps, 250), true_r))
        bayes_rmses_2.append(bpmf.bayes_rmse(islice(samps, 250, None), true_r))
        bayes_rmses_combo.append(bpmf.bayes_rmse(samps, true_r))

        map_rmses.append(bpmf.rmse(true_r))

        print("MAP RMSE:               {}".format(map_rmses[-1]))
        print("Bayes RMSE [first 250]: {}".format(bayes_rmses_1[-1]))
        print("Bayes RMSE [next 250]:  {}".format(bayes_rmses_2[-1]))
        print("Bayes RMSE [combo]:     {}".format(bayes_rmses_combo[-1]))

    from matplotlib import pyplot as plt
    plt.plot(ds, map_rmses, label="MAP")
    plt.plot(ds, bayes_rmses_1, label="Bayes (first 250)")
    plt.plot(ds, bayes_rmses_2, label="Bayes (next 250)")
    plt.plot(ds, bayes_rmses_combo, label="Bayes (all 500)")
    plt.ylabel("RMSE")
    plt.xlabel("Dimensionality")
    plt.legend()
    plt.show()

################################################################################

Key = namedtuple('Key',
        ['nice_name', 'key_fn', 'choose_max', 'wants_pool', 'args'])

KEYS = {
    'random': Key("Random", 'random', True, False, ()),
    'pred-variance': Key("Var[R_ij]", 'pred_variance', True, False, ()),

    'exp-variance': Key("E[Var[R]]", 'exp_variance', False, True, ()),

    'pred': Key("Pred", 'predict', True, False, ()),
    'prob-ge-3.5': Key("Prob >= 3.5", 'prob_ge_cutoff', True, False, (3.5,)),
    'prob-ge-.5': Key("Prob >= .5", 'prob_ge_cutoff', True, False, (.5,)),
    'prob-ge-0': Key("Prob >= 0", 'prob_ge_cutoff', True, False, (0,)),
}

def fetch_samples(bpmf, num, *args, **kwargs):
    try:
        samps = list(islice(bpmf.samples(*args, **kwargs), num))
        pred = bpmf.predict(samps)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    return samps, pred

def full_test(bpmf, samples, real, key_name,
              num_samps=128, lookahead_fit='batch', lookahead_samps=128,
              pool=None, multieval=False, init_rmse=None, test_on=Ellipsis):
    key = KEYS[key_name]
    total = real.size
    picker_fn = getattr(bpmf, key.key_fn)
    chooser = np.argmax if key.choose_max else np.argmin

    if init_rmse is None:
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

            key_kwargs = { 'which': which }
            if key.wants_pool and pool is not None:
                key_kwargs['pool'] = pool

            # XXX: should use lookahead_samps in here somewhere?
            evals = picker_fn(samples, *key.args, **key_kwargs)

            i, j = unrated[:, chooser(evals)]
            vals = bpmf.matrix_results(evals, which)

        bpmf.add_rating(i, j, real[i, j])
        print("{:<40} Queried ({}, {}); {}/{} known".format(
                key.nice_name, i, j, len(bpmf.rated), total))

        samp_args = (bpmf, num_samps)
        samp_kwargs = {'fit_first': True}
        if multieval:
            samples, pred = pool.apply(fetch_samples, samp_args, samp_kwargs)
        else:
            samples, pred = fetch_samples(*samp_args, **samp_kwargs)

        err = rmse(pred[test_on], real[test_on])
        print("{:<40} RMSE {}: {:.5}".format(
            key.nice_name, len(bpmf.rated), err))
        yield len(bpmf.rated), err, (i,j), vals



def compare_active(key_names, latent_d, real, ratings, rating_vals=None,
                   discrete=True, subtract_mean=True, num_steps=None,
                   procs=None, threaded=False,
                   fit_type=('batch',), num_samps=128,
                   test_set='all',
                   **kwargs):
    # figure out which points we know the answers to
    knowable = np.isfinite(real)
    knowable[real == 0] = 0

    # ... and which points we can query
    pickable = knowable.copy()
    pickable[ratings[:,0].astype(int), ratings[:,1].astype(int)] = 0

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
            avail_pts = list(zip(*pickable.nonzero()))
            picked_indices = random.sample(avail_pts, int(test_set))
            picker = np.zeros(pickable.shape, bool)
            picker[tuple(np.transpose(picked_indices))] = 1
        else:
            picker = np.random.binomial(1, test_set, size=pickable.shape)
        test_on = picker * pickable
        query_on = (1 - picker) * pickable

    query_set = set(zip(*query_on.nonzero()))

    print("{} points known, {} to query, testing on {}, {} knowable, {} total"
            .format(ratings.shape[0], query_on.sum(), test_on.sum(),
                    knowable.sum(), real.size))

    # do initial fit
    bpmf_init = BayesianPMF(ratings, latent_d,
            subtract_mean=subtract_mean,
            rating_values=rating_vals,
            discrete_expectations=discrete,
            knowable=query_set,
            fit_type=fit_type)
    print("Doing initial MAP fit...")
    bpmf_init.fit()

    pool = multiprocessing.Pool(procs) if procs is None or procs >= 1 else None

    print("Getting initial MCMC samples...")
    samples = list(islice(bpmf_init.samples(fit_first=fit_type), num_samps))

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
                pool=pool, multieval=threaded,
                num_samps=num_samps,
                init_rmse=init_rmse, test_on=test_on,
                **kwargs)
        results[key_name] = list(islice(res, num_steps))

    if threaded:
        threads = [Thread(name=key_name, target=eval_key, args=(key_name,))
                   for key_name in key_names]
        for thread in threads: thread.start()
        for thread in threads: thread.join()

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

    key_names = KEYS.keys()

    # set up arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--latent-d', '-D', type=int, default=5)
    parser.add_argument('--steps', '-s', type=int, default=None)

    parser.add_argument('--discrete', action='store_true', default=None)
    parser.add_argument('--no-discrete', action='store_false', dest='discrete')

    parser.add_argument('--subtract-mean', action='store_true', default=True)
    parser.add_argument('--no-subtract-mean',
            action='store_false', dest='subtract_mean')

    parser.add_argument('--fit', default='batch')
    parser.add_argument('--lookahead-fit', default='batch')

    parser.add_argument('--samps', '-S', type=int, default=128)
    parser.add_argument('--lookahead-samps', type=int, default=128)

    parser.add_argument('--threaded', action='store_true', default=True)
    parser.add_argument('--unthreaded', action='store_false', dest='threaded')
    parser.add_argument('--procs', '-P', type=int, default=None)

    parser.add_argument('--test-set', default='all')

    parser.add_argument('--load-data', required='True', metavar='FILE')
    parser.add_argument('--save-results', nargs='?', default=True, const=True,
            metavar='FILE')
    parser.add_argument('--no-save-results',
            action='store_false', dest='save_results')

    parser.add_argument('--note', action='append',
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
            data = { '_real': data }

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
                discrete=args.discrete, subtract_mean=args.subtract_mean,
                fit_type=parse_fit_type(args.fit),
                lookahead_fit=args.lookahead_fit,
                num_samps=args.samps, lookahead_samps=args.lookahead_samps,
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
            pickle.dump(results, f)

if __name__ == '__main__':
    main()
