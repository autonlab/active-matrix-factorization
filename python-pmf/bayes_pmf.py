#!/usr/bin/env python3
'''
Implementation of Bayesian PMF (via Gibbs sampling).

Based on Matlab code by Ruslan Salakhutdinov:
http://www.mit.edu/~rsalakhu/BPMF.html
'''

from collections import defaultdict
from copy import deepcopy
from itertools import islice
import multiprocessing
from threading import Thread
import warnings

import numpy as np
from scipy import stats

try:
    from pmf_cy import ProbabilisticMatrixFactorization
except ImportError:
    warnings.warn("cython PMF not available; using pure-python version")
    from pmf import ProbabilisticMatrixFactorization

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
    def __init__(self, rating_tuples, latent_d=5, subtract_mean=True):
        super().__init__(rating_tuples, latent_d=latent_d,
                subtract_mean=subtract_mean)

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
        state = super().__getstate__()
        #state.update(self.__dict__)
        state['__dict__'] = self.__dict__
        return state


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
        norm_ratings = ratings - self.mean_rating

        cov = np.linalg.inv(alpha +
                self.beta * np.dot(rated_feats.T, rated_feats))
        mean = np.dot(cov,
                self.beta * np.dot(rated_feats.T, norm_ratings)
                + np.dot(alpha, mu))

        lam = np.linalg.cholesky(cov)
        return np.dot(lam, np.random.normal(0, 1, self.latent_d)) + mean


    def samples(self, num_gibbs=2, pool=None, multiproc_mode=None,
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

        If fit_first is
            * True or 'batch', first calls .fit() to fit the MAP estimate.
            * a tuple whose first element is 'mini', calls .fit_minibatches()
              with the rest of the tuple as arguments.
            * a tuple whose first element is 'mini-valid', calls
              .fit_minibatches_until_validation() with the rest of the tuple
              as arguments.
            * False, does nothing first.
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
        if fit_first is not False:
            if fit_first is True or fit_first == 'batch':
                fit_first = ('batch',)

            if force_multiproc:
                bpmf = pool.apply(_fit_bpmf, (self,) + fit_first)
                self.users = bpmf.users
                self.items = bpmf.items
            else:
                _fit_bpmf(self, *fit_first)
                self.fit()

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
                        ((self, user_id, True, mu_v, alpha_v, item_sample)
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

    def prob_ge_cutoff(self, samples_iter, cutoff, which=Ellipsis):
        '''
        Gives the portion of the time each matrix element was >= cutoff
        in a series of samples.
        '''
        shape = np.zeros((self.num_users, self.num_items))[which].shape
        counts = np.zeros(shape)
        num = 0
        for u, v in samples_iter:
            counts += self.predicted_matrix(u, v)[which] >= cutoff
            num += 1
        return counts / num

    def random(self, samples_iter, which=Ellipsis):
        shape = np.zeros((self.num_users, self.num_items))[which].shape
        return np.random.rand(*shape)

    def bayes_rmse(self, samples_iter, true_r):
        pred = self.predict(samples_iter)
        return np.sqrt(((true_r - pred)**2).sum() / true_r.size)


# stupid arguments to work around multiprocessing.Pool/pickle silliness
def _hyperparam_sampler(bpmf, *args):
    return bpmf.sample_hyperparam(*args)

def _feat_sampler(args):
    bpmf, *args = args
    return bpmf.sample_feature(*args)

def _fit_bpmf(bpmf, kind, *args, **kwargs):
    if kind == 'batch':
        bpmf.fit(*args, **kwargs)
    elif kind == 'mini':
        bpmf.fit_minibatches(*args, **kwargs)
    elif kind == 'mini-valid':
        bpmf.fit_minibatches_until_validation(*args, **kwargs)
    else:
        raise ValueError("unknown fit type '{}'".format(kind))

    return bpmf


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

KEYS = {
    'random': ("Random", 'random', True),
    'pred-variance': ("Pred Variance", 'pred_variance', True),

    'pred': ("Pred", 'predict', True),
    'prob-ge-3.5': ("Prob >= 3.5", 'prob_ge_cutoff', True, 3.5),
    'prob-ge-.5': ("Prob >= .5", 'prob_ge_cutoff', True, .5),
}

def full_test(bpmf, samples, real, key_name, num_samps,
              pool=None, multiproc_mode=None, fit_type='batch'):

    nice_name, key_fn, choose_max, *key_args = KEYS[key_name]
    total = real.size
    picker_fn = getattr(bpmf, key_fn)
    chooser = np.argmax if choose_max else np.argmin

    yield (len(bpmf.rated), bpmf.bayes_rmse(samples, real), None, None)


    while bpmf.unrated:
        print("{:<40} Picking query point {}...".format(
            nice_name, len(bpmf.rated) + 1))

        if len(bpmf.unrated) == 1:
            vals = None
            i, j = next(iter(bpmf.unrated))
        else:
            # TODO: if multiproc_mode == 'force'...?
            unrated = np.array(list(bpmf.unrated)).T
            which = tuple(unrated)
            evals = picker_fn(samples, *key_args, which=which)
            i, j = unrated[:, chooser(evals)]
            vals = bpmf.matrix_results(evals, which)

        bpmf.add_rating(i, j, real[i, j])
        print("{:<40} Queried ({}, {}); {}/{} known".format(
                nice_name, i, j, len(bpmf.rated), total))

        samp_gen = bpmf.samples(pool=pool, multiproc_mode=multiproc_mode,
                                fit_first=fit_type)
        samples = list(islice(samp_gen, num_samps))

        rmse = bpmf.bayes_rmse(samples, real)
        print("{:<40} RMSE {}: {:.5}".format(nice_name, len(bpmf.rated), rmse))
        yield len(bpmf.rated), rmse, (i,j), vals



def compare_active(key_names, latent_d, real, ratings, rating_vals=None,
                   num_samps=128, num_steps=None, procs=None, threaded=False):
    # do initial fit
    bpmf_init = BayesianPMF(ratings, latent_d)
    print("Doing initial MAP fit...")
    bpmf_init.fit()

    pool = multiprocessing.Pool(procs) if procs is None or procs >= 1 else None

    print("Getting initial MCMC samples...")
    samples = list(islice(bpmf_init.samples(pool=pool), num_samps))
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
                num_samps, pool, 'force' if threaded else None)
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
    parser.add_argument('--samps', '-S', type=int, default=128)

    parser.add_argument('--threaded', action='store_true', default=True)
    parser.add_argument('--unthreaded', action='store_false', dest='threaded')
    parser.add_argument('--procs', '-P', type=int, default=None)

    parser.add_argument('--load-data', required='True', metavar='FILE')
    parser.add_argument('--save-results', nargs='?', default=True, const=True,
            metavar='FILE')
    parser.add_argument('--no-save-results',
            action='store_false', dest='save_results')

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

    # do the comparison
    try:
        results = compare_active(
                key_names=args.keys,
                latent_d=args.latent_d,
                real=real, ratings=ratings, rating_vals=rating_vals,
                num_samps=args.samps, num_steps=args.steps,
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
