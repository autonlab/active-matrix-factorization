#!/usr/bin/env python3
'''
Implementation of Bayesian PMF (via Gibbs sampling).

Based on Matlab code by Ruslan Salakhutdinov:
http://www.mit.edu/~rsalakhu/BPMF.html
'''

from collections import defaultdict
from itertools import islice
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
    for count, x in enumerate(i):
        total += x
    return total / (count + 2)

################################################################################

class BayesianPMF(ProbabilisticMatrixFactorization):
    def __init__(self, rating_tuples, latent_d=5):
        super().__init__(rating_tuples, latent_d)

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


    def samples(self, num_gibbs=2):
        '''
        Runs the Markov chain starting from the current MAP approximation in
        self.users, self.items. Yields sampled user, item features forever.

        Note that it actually just yields the same numpy arrays over and over:
        if you need access to the samples later, make a copy. That is, don't do
            samps = list(islice(self.samples(), n))
        Instead, do
            samps = [(u.copy(), v.copy()) for u, v in islice(self.samples(), n)]

        If you add ratings after starting this iterator, it'll continue on
        without accounting for them.

        Does num_gibbs updates after each hyperparameter update, then yields
        the result.
        '''
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


        # initialize the Markov chain with the current MAP estimate
        # TODO: MAP search doesn't currently normalize by the mean rating
        #       should do that, or there'll be a while for burn-in to adjust
        user_sample = self.users.copy()
        item_sample = self.items.copy()

        # mu is the average value for each latent dimension
        mu_u = np.mean(user_sample, axis=0).T
        mu_v = np.mean(item_sample, axis=0).T

        # alpha is the inverse covariance among latent dimensions
        alpha_u = np.linalg.inv(np.cov(user_sample, rowvar=0))
        alpha_v = np.linalg.inv(np.cov(item_sample, rowvar=0))


        while True:
            # sample from hyperparameters
            mu_u, alpha_u = self.sample_hyperparam(user_sample, True)
            mu_v, alpha_v = self.sample_hyperparam(item_sample, False)

            # Gibbs updates for user, item feature vectors
            # TODO: parallelize

            for gibbs in range(num_gibbs):
                #print('\t\t Gibbs sampling {}'.format(gibbs))

                for user_id in range(self.num_users):
                    #print('user {}'.format(user_id))

                    user_sample[user_id, :] = self.sample_feature(
                            user_id, True, mu_v, alpha_v, item_sample,
                            *items_by_user[user_id])


                for item_id in range(self.num_items):
                    #print('item {}'.format(item_id))

                    item_sample[item_id, :] = self.sample_feature(
                            item_id, False, mu_v, alpha_v, user_sample,
                            *users_by_item[item_id])

            yield user_sample, item_sample

    def predict(self, samples_iter):
        # TODO: cut off the prediction here?
        return iter_mean(np.dot(u, v.T) + self.mean_rating
                         for u, v in samples_iter)

if __name__ == '__main__':
    from pmf import fake_ratings

    ratings, true_u, true_v = fake_ratings(noise=1)
    true_r = np.dot(true_u, true_v.T)

    ds = [5, 8, 10, 12]
    map_rmses = []
    bayes_rmses = []

    for latent_d in ds:
        bpmf = BayesianPMF(ratings, latent_d)

        print("fitting MAP ({})".format(latent_d))
        for ll in bpmf.fit_lls():
            pass
            #print("LL {}".format(ll))

        predicted_map = bpmf.predicted_matrix()

        samples = bpmf.samples()
        def counter():
            n = 0
            def inner():
                nonlocal n
                n += 1
                return n
            return inner
        n = counter()

        def printslice(count):
            for i in range(count):
                #print("sample {}".format(n()))
                yield next(samples)

        def rmse(count):
            pred = bpmf.predict(printslice(count))
            return np.sqrt(((true_r - pred)**2).sum() / true_r.size)

        print("doing MCMC ({})".format(latent_d))
        b_rmse = rmse(200)
        bayes_rmses.append(b_rmse)

        map_rmse = bpmf.rmse(true_r)
        map_rmses.append(map_rmse)

        print()
        print("MAP RMSE: {}".format(map_rmse))
        print("Bayes RMSE [250]: {}".format(b_rmse))

    from matplotlib import pyplot as plt
    plt.plot(ds, map_rmses, name="MAP")
    plt.plot(ds, bayes_rmses, name="Bayes")
