#!/usr/bin/env python
'''
Code to do active learning on a PMF model.
'''

from __future__ import division

from copy import deepcopy
import itertools as itools
import math
import operator
import random

import numpy as np

from pmf import ProbabilisticMatrixFactorization

################################################################################
### Helpers

def tripexpect(mean, cov, a, b, c):
    '''E[X_a X_b X_c] for N(mean, cov)'''
    return mean[a] * mean[b] * mean[c] + \
            mean[a]*cov[b,c] + mean[b]*cov[a,c] + mean[c]*cov[a,b]

def quadexpect(mean, cov, a, b, c, d):
    '''E[X_a X_b X_c X_d] for N(mean, cov) and distinct a/b/c/d'''
    # TODO: try method of Kan (2008), see if it's faster?
    abcd = [a, b, c, d]
    ma, mb, mc, md = mean[abcd]

    return (
        # product of the means
        ma * mb * mc * md

        # pairs of means times cov of the other two
        + ma * mb * cov[c,d]
        + ma * mc * cov[b,d]
        + ma * md * cov[b,c]
        + mb * mc * cov[a,d]
        + mb * md * cov[a,c]
        + mc * md * cov[a,b]

        # pairs of covariances (Isserlis)
        + cov[a,b] * cov[c,d]
        + cov[a,c] * cov[b,d]
        + cov[a,d] * cov[b,c]
    )

def exp_squared(mean, cov, a, b):
    '''E[X_a^2 X_b^2] for N(mean, cov)'''
    return 4 * mean[a] * mean[b] * cov[a,b] + 2*cov[a,b]**2 + \
            (mean[a]**2 + cov[a,a]) * (mean[b]**2 + cov[b,b])


def project_psd(mat, min_eig=0):
    '''
    Project a real symmetric matrix to PSD by discarding any negative
    eigenvalues from its spectrum. Passing min_eig > 0 lets you similarly make
    it positive-definite, though this may not technically be a projection...?

    Symmetrizes the matrix before projecting.
    '''
    #TODO: better way to project to strictly positive definite?
    mat = (mat + mat.T) / 2
    vals, vecs = np.linalg.eigh(mat)
    if vals.min() < min_eig:
        mat = np.dot(vecs, np.dot(np.diag(np.maximum(vals, min_eig)), vecs.T))
        mat = (mat + mat.T) / 2
    return mat


################################################################################
### Main code

# helper for multiprocessing
class ActivePMFEvaluator(object):
    def __init__(self, apmf, key):
        self.apmf = apmf
        self.key_name = key.__name__

    def __call__(self, ij):
        return getattr(self.apmf, self.key_name)(ij)


class ActivePMF(ProbabilisticMatrixFactorization):
    def __init__(self, rating_tuples, latent_d=1):
        # the actual PMF model
        super(ActivePMF, self).__init__(rating_tuples, latent_d)

        # parameters of the normal approximation
        self.mean = None
        self.cov = None

        n = self.num_users
        m = self.num_items
        d = self.latent_d

        self.approx_dim = k = (n + m) * d
        self.num_params = k + k * (k+1) / 2 # means and covariances

        # indices into the normal approx for the users / items arrays
        # e.g. mean[u[k,i]] corresponds to the mean for U_{ki}
        self.u = np.arange(0, n * d).reshape(n, d).T
        self.v = np.arange(n * d, (n+m) * d).reshape(m, d).T

        # training options for the normal approximation
        self.normal_learning_rate = 1e-4
        self.min_eig = 1e-5 # minimum eigenvalue to be considered positive-def


    def initialize_approx(self):
        '''
        Also sets up the normal approximation to be near the PMF result
        (throwing away any old approximation).
        '''
        # set mean to PMF's MAP values
        self.mean = np.hstack((self.users.reshape(-1), self.items.reshape(-1)))

        # set covariance to a random positive-definite matrix
        s = np.random.normal(0, 2, (self.approx_dim, self.approx_dim))
        self.cov = project_psd(s, min_eig=self.min_eig)

    def mean_meandiff(self):
        p = np.hstack((self.users.reshape(-1), self.items.reshape(-1)))
        return np.abs(self.mean - p).mean()

    def _exp_dotprod_sq(self, i, j, mean, cov):
        '''E[ (U_i^T V_j)^2 ]'''
        u = self.u
        v = self.v

        # TODO: vectorize as much as possible
        exp = 0
        for k in xrange(self.latent_d):
            uki = u[k,i]
            vkj = v[k,j]

            exp += exp_squared(mean, cov, uki, vkj)

            for l in xrange(k+1, self.latent_d):
                exp += 2 * quadexpect(mean, cov, uki, vkj, u[l,i], v[l,j])
        return exp

    def kl_divergence(self, mean=None, cov=None):
        '''KL(PMF || approximation), up to an additive constant'''
        if mean is None: mean = self.mean
        if cov is None: cov = self.cov
        if mean is None or cov is None:
            raise ValueError("run initialize_approx first")

        u = self.u
        v = self.v

        us = u.reshape(-1)
        vs = v.reshape(-1)

        # terms based on the squared error
        div = (
            sum(
                # E[ (U_i^T V_j)^2 ]
                self._exp_dotprod_sq(i, j, mean, cov)

                # - 2 R_ij E[U_i^T V_j]
                - 2 * rating *
                    (mean[u[:,i]] * mean[v[:,j]] + cov[u[:,i], v[:,j]]).sum()

                for i, j, rating in self.ratings)
            + (self.ratings[:,2] ** 2).sum() # sum (R_ij^2)
        ) / (2 * self.sigma_sq)

        # regularization terms
        # cov[us, us] only gives us diagonal terms, unlike in matlab
        div += ((mean[us]**2).sum() + cov[us, us].sum()) / (2*self.sigma_u_sq)
        div += ((mean[vs]**2).sum() + cov[vs, vs].sum()) / (2*self.sigma_v_sq)

        # entropy term
        det_sign, log_det = np.linalg.slogdet(cov)
        div += log_det / 2

        return div

    def normal_gradient(self, mean=None, cov=None):
        if mean is None: mean = self.mean
        if cov is None: cov = self.cov
        if mean is None or cov is None:
            raise ValueError("run initialize_approx first")

        u = self.u
        v = self.v

        us = u.reshape(-1)
        vs = v.reshape(-1)

        sig = self.sigma_sq

        # note that we're actually only differentiating by one triangular half
        # of the covariance matrix, but we make grad_cov the full square anyway
        grad_mean = np.zeros_like(mean)
        grad_cov = np.zeros_like(cov)

        def inc_cov_quadexp_grad(a,b, c,d):
            inc = np.sum(mean[c] * mean[d] + cov[c, d]) / sig
            grad_cov[a, b] += inc
            grad_cov[b, a] += inc

        for i, j, rating in self.ratings:
            # gradient of sum_k sum_{l>k} E[ U_ki V_kj U_li V_lj ] / sigma^2
            # (doubled because of symmetry, cancels with 2 in denom)
            for k in xrange(self.latent_d-1):
                uki = u[k, i]
                vkj = v[k, j]
                # vectorize out our sums over l
                uli = u[k+1:, i]
                vlj = v[k+1:, j]

                grad_mean[uki] += tripexpect(mean,cov, vkj,uli,vlj).sum() / sig
                grad_mean[vkj] += tripexpect(mean,cov, uki,uli,vlj).sum() / sig
                grad_mean[uli] += tripexpect(mean,cov, uki,vkj,vlj).sum() / sig
                grad_mean[vlj] += tripexpect(mean,cov, uki,vkj,uli).sum() / sig

                inc_cov_quadexp_grad(uki,vkj, uli,vlj)
                inc_cov_quadexp_grad(uki,uli, vkj,vlj)
                inc_cov_quadexp_grad(uki,vlj, vkj,uli)
                inc_cov_quadexp_grad(vkj,uli, uki,vlj)
                inc_cov_quadexp_grad(vkj,vlj, uki,uli)
                inc_cov_quadexp_grad(uli,vlj, uki,vkj)

            # everything else can just be vectorized over k
            uki = u[:,i]
            vkj = v[:,j]

            muki = mean[uki]
            mvkj = mean[vkj]

            # gradient of \sum_k E[ U_ki^2 V_kj^2 ] / (2 sigma^2)
            grad_mean[uki] += (2 * mvkj * cov[uki,vkj]
                               + muki * (mvkj**2 + cov[vkj,vkj])) / sig
            grad_mean[vkj] += (2 * muki * cov[uki,vkj]
                               + mvkj * (muki**2 + cov[uki,uki])) / sig

            grad_cov[uki,uki] += (mvkj**2 + cov[vkj,vkj]) / (2*sig)
            grad_cov[vkj,vkj] += (muki**2 + cov[uki,uki]) / (2*sig)

            inc = 2 * (muki * mvkj + cov[uki,vkj]) / sig
            grad_cov[uki,vkj] += inc
            grad_cov[vkj,uki] += inc

            # gradient of - \sum_k Rij E[U_ki V_kj] / sigma^2
            grad_mean[uki] -= mvkj * (rating / sig)
            grad_mean[vkj] -= muki * (rating / sig)

            grad_cov[uki,vkj] -= rating / sig
            grad_cov[vkj,uki] -= rating / sig


        # gradient of \sum_i \sum_k E[U_ki^2] / (2 sigma_u^2), same for V
        grad_mean[us] += mean[us] / self.sigma_u_sq
        grad_mean[vs] += mean[vs] / self.sigma_v_sq

        grad_cov[us,us] += 1 / (2 * self.sigma_u_sq)
        grad_cov[vs,vs] += 1 / (2 * self.sigma_v_sq)

        # gradient of -ln(|cov|)/2
        # need each cofactor of the matrix divided by its determinant;
        # this is just the transpose of its inverse
        grad_cov += np.linalg.inv(cov).T / 2

        return grad_mean, grad_cov


    def fit_normal(self):
        '''
        Fit the multivariate normal over the elements of U and V that
        best approximates the distribution defined by the current PMF model,
        using gradient descent.
        '''
        for kl in self.fit_normal_kls():
            pass


    def fit_normal_kls(self):
        '''
        Find the best normal approximation of the PMF model, yielding the
        current KL divergence at each step.
        '''
        lr = self.normal_learning_rate
        old_kl = self.kl_divergence()
        converged = False

        while not converged:
            grad_mean, grad_cov = self.normal_gradient()

            # take one step, trying different learning rates if necessary
            while True:
                #print "  setting learning rate =", lr
                new_mean = self.mean - lr * grad_mean
                new_cov = project_psd(self.cov - lr * grad_cov,
                                      min_eig=self.min_eig)
                new_kl = self.kl_divergence(new_mean, new_cov)

                # TODO: configurable momentum, stopping conditions
                if new_kl < old_kl:
                    self.mean = new_mean
                    self.cov = new_cov
                    lr *= 1.25

                    if old_kl - new_kl < .005:
                        converged = True
                    break
                else:
                    lr *= .5

                    if lr < 1e-10:
                        converged = True
                        break

            yield new_kl
            old_kl = new_kl


    def approx_pred_means_vars(self):
        '''
        Returns the mean and variance of the predicted R matrix under
        the normal approximation.
        '''
        p_mean = np.zeros((self.num_users, self.num_items))
        p_var = np.zeros((self.num_users, self.num_items))

        mean = self.mean
        cov = self.cov

        for i in xrange(self.num_users):
            us = self.u[:, i]
            for j in xrange(self.num_items):
                vs = self.v[:, j]
                p_mean[i, j] = m = (mean[us] * mean[vs] + cov[us, vs]).sum()
                p_var[i, j] = self._exp_dotprod_sq(i, j, mean, cov) - m**2

        return p_mean, p_var


    def pred_variance(self, ij):
        '''
        The variance in our prediction for R_ij, according to the current
        approximation.
        '''
        i, j = ij

        mean = self.mean
        cov = self.cov
        us = self.u[:, i]
        vs = self.v[:, j]

        return (
            # E[ (U_i^T V_j)^2 ]
            self._exp_dotprod_sq(i, j, mean, cov)

            # - E[U_i^T V_j] ^ 2
            - (mean[us] * mean[vs] + cov[us, vs]).sum() ** 2
        )

    def random_weighting(self, ij):
        return random.random()

    def exp_approx_entropy_byapprox(self, ij):
        return self.exp_approx_entropy(ij, False)

    def exp_approx_entropy(self, ij, use_pmf=True):
        '''
        The expected entropy in our approximation of U and V if we know Rij,
        calculated according to our current belief about the distribution of
        Rij, up to a constant common to all (i,j) pairs.
        '''
        from scipy.integrate import quad
        i, j = ij

        # TODO: use PMF's distribution for Rij or the normal approximation's?
        if use_pmf:
            mean = np.dot(self.users[i,:], self.items[j,:])
            var = self.sigma_sq
        else:
            # TODO: this isn't actually right (using the normal distribution
            # with matching mean/variance instead of the actual, unknown
            # distribution)
            us = self.u[:, i]
            vs = self.v[:, j]

            mean = (self.mean[us] * self.mean[vs] + self.cov[us, vs]).sum()
            var = self._exp_dotprod_sq(i, j, self.mean, self.cov) - mean**2

        std = math.sqrt(var)
        scale = math.sqrt(2 * math.pi) * std

        def weighted_entropy_with_ij_val(v):
            assert isinstance(v, float)
            apmf = deepcopy(self)
            apmf.add_rating(i, j, v)
            apmf.fit() # XXX is this necessary?
            apmf.fit_normal()

            sign, logdet = np.linalg.slogdet(apmf.cov)
            assert sign == 1
            return -logdet * np.exp(0.5 * (v - mean)**2 / var) / scale

        # only take the expectation out to 1.96 sigma (95% of normal mass)
        left = mean - 1.96 * std
        right = mean + 1.96 * std

        est, abserr = quad(weighted_entropy_with_ij_val, left, right,
                epsabs=1e-2, epsrel=1e-2)
        print "(%d, %d) integrated from %g to %g:\t %g +- %g" % (
                i, j, left, right, est, abserr)
        return est


    def pick_query_point(self, pool=None, key=None):
        return self.pick_query_point_multiprocessing(pool, key)

    def pick_query_point_single(self, pool=None, key=None):
        '''
        Use the approximation of the PMF model to select the next point to
        query, choosing elements according to key, which should be a function
        that takes an ActivePMF instance and a user-item pair as arguments.
        Defaults to pred_variance.

        The choices can be limited to an iterable pool (default self.unrated).
        '''
        if pool is None:
            pool = self.unrated
        if key is None:
            key = ActivePMF.pred_variance
        return max(pool, key=lambda ij: key(self, ij))

    def pick_query_point_multiprocessing(self, pool=None, key=None, procs=None):
        # NOTE: key has to be a method of ActivePMF
        if pool is None:
            pool = self.unrated
        if key is None:
            key = ActivePMF.pred_variance

        # TODO: use np.save instead of pickle to transfer data
        # (or maybe shared memory? http://stackoverflow.com/q/5033799/344821)
        from multiprocessing import Pool
        processes = Pool(procs)

        vals = processes.map(ActivePMFEvaluator(self, key), pool)
        return max(zip(pool, vals), key=operator.itemgetter(1))[0]



################################################################################
### Testing code

def make_fake_data(noise=.25, num_users=10, num_items=10,
                        rating_prob=0, rank=5):
    u = np.random.normal(0, 2, (num_users, rank))
    v = np.random.normal(0, 2, (num_items, rank))

    ratings = np.dot(u, v.T) + \
            np.random.normal(0, noise, (num_users, num_items))

    mask = np.random.binomial(1, rating_prob, ratings.shape)

    # make sure every row/col has at least one rating
    for zero_col in np.logical_not(mask.sum(axis=0)).nonzero()[0]:
        mask[random.randrange(num_users), zero_col] = 1

    for zero_row in np.logical_not(mask.sum(axis=1)).nonzero()[0]:
        mask[zero_row, random.randrange(num_items)] = 1

    assert (mask.sum(axis=0) > 0).all()
    assert (mask.sum(axis=1) > 0).all()

    # convert into the list-of-ratings form we want
    rates = np.zeros((mask.sum(), 3))
    for idx, (i, j) in enumerate(np.transpose(mask.nonzero())):
        rates[idx] = [i, j, ratings[i, j]]

    return ratings, rates


def plot_variances(apmf, vmax=None):
    from matplotlib import pyplot as plt
    var = np.zeros((apmf.num_users, apmf.num_items))
    total = 0
    for i, j in itools.product(xrange(apmf.num_users), xrange(apmf.num_items)):
        if (i, j) in apmf.rated:
            var[i, j] = float('nan')
        else:
            var[i, j] = apmf.pred_variance(i, j)
            total += var[i,j]

    plt.imshow(var.T, interpolation='nearest', origin='lower',
            norm=plt.Normalize(0, vmax))
    plt.xlabel("User")
    plt.ylabel("Item")
    plt.title("Prediction Variances; total = %g" % total)
    plt.colorbar()

    return var

def plot_predictions(apmf, real):
    from matplotlib import pyplot as plt

    pred = apmf.predicted_matrix()
    a_mean, a_var = apmf.approx_pred_means_vars()
    a_std = np.sqrt(a_var)

    xs = (real, pred, a_mean)
    norm = plt.Normalize(min(a.min() for a in xs), max(a.max() for a in xs))

    plt.subplot(2, 2, 1)
    plt.imshow(real, norm=norm, interpolation='nearest')
    plt.title("Real")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(pred, norm=norm, interpolation='nearest')
    plt.title("Predicted")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(a_mean, norm=norm, interpolation='nearest')
    plt.title("Approximation: Means")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(np.sqrt(a_std), interpolation='nearest')
    plt.title("Approximation: Std Devs")
    plt.colorbar()


def full_test(apmf, real, picker_key=ActivePMF.pred_variance,
              fit_normal=True, processes=None):
    print "Training PMF:"
    for ll in apmf.fit_lls():
        print "\tLL: %g" % ll

    apmf.initialize_approx()

    if fit_normal:
        print "Fitting normal:"
        for kl in apmf.fit_normal_kls():
            print "\tKL: %g" % kl
            assert kl > -1e5

        print "Mean diff of means: %g; mean cov %g" % (
                apmf.mean_meandiff(), np.abs(apmf.cov.mean()))

    total = apmf.num_users * apmf.num_items
    rmse = apmf.rmse(real)
    print "RMSE: %g" % rmse
    yield len(apmf.rated), rmse


    while apmf.unrated:
        print
        #print '=' * 80

        if processes == 1:
            i, j = apmf.pick_query_point_single(key=picker_key)
        else:
            i, j = apmf.pick_query_point_multiprocessing(
                    key=picker_key, procs=processes)

        apmf.add_rating(i, j, real[i, j])
        print "Queried (%d, %d); %d/%d known" % (i, j, len(apmf.rated), total)

        print "Training PMF:"
        for ll in apmf.fit_lls():
            print "\tLL: %g" % ll

        if fit_normal:
            print "Fitting normal:"
            for kl in apmf.fit_normal_kls():
                print "\tKL: %g" % kl
                assert kl > -1e5

            print "Mean diff of means: %g; mean cov %g" % (
                    apmf.mean_meandiff(), np.abs(apmf.cov.mean()))

        rmse = apmf.rmse(real)
        print "RMSE: %g" % rmse
        yield len(apmf.rated), rmse


KEY_OPTIONS = {
    "uv-entropy-approx": (
        "U/V Entropy (Normal Prediction)",
        ActivePMF.exp_approx_entropy_byapprox,
        True),
    "uv-entropy-pred": ("U/V Entropy (PMF Prediction)",
        ActivePMF.exp_approx_entropy,
        True),
    "pred-variance": ("Prediction Variance", ActivePMF.pred_variance, True),
    "random": ("Random", ActivePMF.random_weighting, False),
}

def compare(key_names, plot=True, saveplot=None, latent_d=5,
            processes=None, **kwargs):
    real, ratings = make_fake_data(**kwargs)
    apmf = ActivePMF(ratings, latent_d=latent_d)

    keys = [KEY_OPTIONS[k] for k in key_names]

    results = []
    for name, key, do_fit in keys:
        print '=' * 80
        print "Starting", name
        print '=' * 80
        results.append(list(full_test(deepcopy(apmf), real, key, do_fit,
            processes=processes)))
        print '=' * 80
        print '=' * 80

    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.xlabel("# of rated elements")
        plt.ylabel("RMSE")

        for (name, key, do_fit), result in zip(keys, results):
            plt.plot(*zip(*result), label=name)

        plt.legend()
        if saveplot is None:
            plt.show()
        else:
            plt.savefig(saveplot)


def main():
    key_names = set(KEY_OPTIONS.keys())

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-d', '-D', type=int, default=5)
    parser.add_argument('--gen-rank', '-R', type=int, default=5)
    parser.add_argument('--noise', type=float, default=.25)
    parser.add_argument('--num-users', '-N', type=int, default=10)
    parser.add_argument('--num-items', '-M', type=int, default=10)
    parser.add_argument('--rating-prob', '-r', type=float, default=0)
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', action='store_false', dest='plot')
    parser.add_argument('--processes', '-P', type=int, default=None)
    parser.add_argument('--outfile', default=None)
    parser.add_argument('keys', nargs='*', default=list(key_names))
    args = parser.parse_args()

    for k in args.keys:
        if k not in key_names:
            import sys
            sys.stderr.write("Invalid key name %s; options are %s.\n" % (
                k, ', '.join(key_names)))
            sys.exit(1)

    try:
        compare(args.keys or key_names,
                num_users=args.num_users, num_items=args.num_items,
                rank=args.gen_rank, latent_d=args.latent_d,
                noise=args.noise,
                rating_prob=args.rating_prob,
                plot=args.plot, saveplot=args.outfile,
                processes=args.processes)
    except Exception:
        import pdb, traceback
        print
        traceback.print_exc()
        print
        pdb.post_mortem()

if __name__ == '__main__':
    main()
