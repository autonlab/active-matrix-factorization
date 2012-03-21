#!/usr/bin/env python
'''
Code to do active learning on a PMF model.
'''

from __future__ import division

from copy import deepcopy
import functools
from itertools import product
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

def exp_a2bc(mean, cov, a, b, c):
    '''E[X_a^2 X_b X_c for N(mean, cov)'''
    ma, mb, mc = mean[[a,b,c]]
    return (
        (ma**2 + cov[a,a]) * (mb * mc + cov[b,c])
        + 2 * ma * mc * cov[a,b]
        + 2 * ma * mb * cov[a,c]
        + 2 * cov[a,b] * cov[a,c]
    )


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
### Helpers

# avoid pickling methods
class ActivePMFEvaluator(object):
    def __init__(self, apmf, key):
        self.apmf = apmf
        self.key_name = key.__name__

    def __call__(self, ij):
        return getattr(self.apmf, self.key_name)(ij)

# decorators to set properties of different active learning criteria
def do_normal_fit(val):
    def decorator(f):
        f.do_normal_fit = val
        return f
    return decorator

def spawn_processes(val):
    def decorator(f):
        f.spawn_processes = val
        return f
    return decorator

def nice_name(name):
    def decorator(f):
        f.nice_name = name
        return f
    return decorator

def minimize(f):
    f.chooser = min
    return f
def maximize(f):
    f.chooser = max
    return f


################################################################################
### Main code

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

    ############################################################################
    ### Gradient descent to find the normal approximation

    def initialize_approx(self):
        '''
        Sets up the normal approximation to be near the MAP PMF result
        (throwing away any old approximation).
        '''
        # set mean to PMF's MAP values
        self.mean = np.hstack((self.users.reshape(-1), self.items.reshape(-1)))

        # set covariance to a random positive-definite matrix
        s = np.random.normal(0, 2, (self.approx_dim, self.approx_dim))
        self.cov = project_psd(s, min_eig=self.min_eig)

    def kl_divergence(self, mean=None, cov=None):
        '''KL(PMF model || approximation), up to an additive constant'''
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


    ############################################################################
    ### Helpers to get various things based on the current approximation

    def mean_meandiff(self):
        '''
        Mean absolute difference between the means of the normal approximation
        and the MAP values.
        '''
        p = np.hstack((self.users.reshape(-1), self.items.reshape(-1)))
        return np.abs(self.mean - p).mean()

    def approx_pred_means_vars(self):
        '''
        Returns the mean and variance of the predicted R matrix under
        the normal approximation.
        '''
        p_mean = np.zeros((self.num_users, self.num_items))
        p_var = np.zeros((self.num_users, self.num_items))

        mean = self.mean
        cov = self.cov

        # TODO: vectorize
        for i in xrange(self.num_users):
            us = self.u[:, i]
            for j in xrange(self.num_items):
                vs = self.v[:, j]
                p_mean[i, j] = m = (mean[us] * mean[vs] + cov[us, vs]).sum()
                p_var[i, j] = self._exp_dotprod_sq(i, j, mean, cov) - m**2

        return p_mean, p_var

    def approx_pred_covs(self):
        '''
        Returns the covariance between elements of the predicted R matrix
        under the normal approximation.
        '''
        ijs = list(product(xrange(self.num_users), xrange(self.num_items)))
        pred_covs = np.zeros((len(ijs), len(ijs)))
        # covariance of U_i.V_j with U_a.V_b

        mean = self.mean
        cov = self.cov

        qexp = functools.partial(quadexpect, mean, cov)
        a2bc = functools.partial(exp_a2bc, mean, cov)

        # TODO: vectorize
        for idx1, (i, j) in enumerate(ijs):
            u_i = self.u[:,i]
            v_j = self.v[:,j]

            for idx2, (a, b) in enumerate(ijs):
                if idx1 == idx2: # variance of U_i.V_j
                    m = (mean[u_i] * mean[v_j] + cov[u_i, v_j]).sum()
                    cv = self._exp_dotprod_sq(i, j, mean, cov) - m**2

                elif i == a: # cov of U_i.V_j with U_i.V_b, j != b
                    v_b = self.v[:, b]
                    cv = 0

                    # sum_{k,l} E[Uki Vkj Uli Vlb]
                    for k in xrange(self.latent_d):
                        for l in xrange(k):
                            cv += qexp(u_i[k], v_j[k], u_i[l], v_b[l])

                        cv += a2bc(u_i[k], v_j[k], v_b[k])

                        for l in xrange(k+1, self.latent_d):
                            cv += qexp(u_i[k], v_j[k], u_i[l], v_b[l])

                    # - sum_{k,l} E[Uki Vkj] E[Uli Vlb]
                    e_ij = mean[u_i] * mean[v_j] + cov[u_i, v_j]
                    e_ib = mean[u_i] * mean[v_b] + cov[u_i, v_b]
                    cv -= sum(ij_k * ib_l for ij_k, ib_l in product(e_ij, e_ib))

                elif j == b: # cov of U_i.V_j with U_a.V_j, i != a
                    u_a = self.u[:, a]
                    cv = 0

                    # sum_{k,l} E[Uki Vkj Ula Vlj]
                    for k in xrange(self.latent_d):
                        for l in xrange(k):
                            cv += qexp(u_i[k], v_j[k], u_a[l], v_j[l])

                        cv += a2bc(v_j[k], u_i[k], u_a[k])

                        for l in xrange(k+1, self.latent_d):
                            cv += qexp(u_i[k], v_j[k], u_a[l], v_j[l])

                    # - sum_{k,l} E[Uki Vkj] E[Ula Vlj]
                    e_ij = mean[u_i] * mean[v_j] + cov[u_i, v_j]
                    e_aj = mean[u_a] * mean[v_j] + cov[u_a, v_j]
                    cv -= sum(ij_k * aj_l for ij_k, aj_l in product(e_ij, e_aj))

                else: # cov of U_i.V_j with U_a.V_b, i != a, j != b
                    u_a = self.u[:, a]
                    v_b = self.v[:, b]
                    cv = 0

                    # sum_{k,l} E[Uki Vkj Ula Vlb]
                    for k in xrange(self.latent_d):
                        for l in xrange(self.latent_d):
                            cv += qexp(u_i[k], v_j[k], u_a[l], v_b[l])

                    # - sum_{k,l} E[Uki Vkj] E[Ula Vlb]
                    e_ij = mean[u_i] * mean[v_j] + cov[u_i, v_j]
                    e_ab = mean[u_a] * mean[v_b] + cov[u_a, v_b]
                    cv -= sum(ij_k * ab_l for ij_k, ab_l in product(e_ij, e_ab))

                pred_covs[idx1, idx2] = cv

        return pred_covs


    ############################################################################
    ### Various criteria to use to pick query points

    @do_normal_fit(False)
    @spawn_processes(False)
    @nice_name("Random")
    @maximize
    def random_weighting(self, ij):
        return random.random()

    @do_normal_fit(True)
    @spawn_processes(False)
    @nice_name("Prediction Variance")
    @maximize
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

    def _approx_entropy(self):
        '''Entropy of the normal approximation, up to an additive constant'''
        sign, logdet = np.linalg.slogdet(self.cov)
        assert sign == 1
        return logdet

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("U/V Expected Entropy (MAP)")
    @minimize
    def exp_approx_entropy(self, ij):
        '''
        The expected entropy in our approximation of U and V if we know Rij,
        calculated according to our current MAP belief about the distribution
        of Rij, up to a constant common to all (i,j) pairs.
        '''
        return self._integrate_prediction(ij, ActivePMF._approx_entropy,
                use_map=True)

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("U/V Expected Entropy (Approximation)")
    @minimize
    def exp_approx_entropy_byapprox(self, ij):
        '''
        The expected entropy in our approximation of U and V if we know Rij,
        calculated according to our current belief about the distribution of
        Rij pretending that it's normal with mean and variance defined by our
        distribution over U and V, up to a constant common to all (i,j) pairs.
        '''
        return self._integrate_prediction(ij, ActivePMF._approx_entropy,
                use_map=False)


    def _pred_entropy_bound(self):
        '''
        Upper bound on the entropy of the predicted matrix, up to an additive
        constant.
        '''
        p_cov = self.approx_pred_covs()
        sign, logdet = np.linalg.slogdet(p_cov)
        assert sign == 1
        return logdet

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("Prediction Expected Entropy Bound (MAP)")
    @minimize
    def exp_pred_entropy_bound(self, ij):
        '''
        The expectation on an upper bound of the entropy in our predicted R
        matrix if we know Rij, calculated according to our current MAP belief
        about the distribution of Rij, up to a constant common to all (i,j)
        pairs.
        '''
        return self._integrate_prediction(ij, ActivePMF._pred_entropy_bound,
                use_map=True)

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("Prediction Expected Entropy Bound (Approximation)")
    @minimize
    def exp_pred_entropy_bound_byapprox(self, ij):
        '''
        The expectation on an upper bound of the entropy in our predicted R
        matrix if we know Rij, calculated according to our current belief about
        the distribution of Rij pretending that it's normal with mean and
        variance defined by our distribution over U and V, up to a constant
        common to all (i,j) pairs.
        '''
        return self._integrate_prediction(ij, ActivePMF._pred_entropy_bound,
                use_map=False)


    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("Expected Total Prediction Variance (MAP)")
    @minimize
    def exp_total_variance(self, ij):
        '''
        The total expected variance in our predicted R matrix if we know Rij,
        calculated according to our current MAP belief about the distribution
        of Rij, up to a constant common to all (i,j) pairs.
        '''
        return self._integrate_prediction(ij,
                lambda self: self.approx_pred_means_vars()[1].sum(),
                use_map=True)

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("Expected Total Prediction Variance (Approximation)")
    @minimize
    def exp_total_variance_byapprox(self, ij):
        '''
        The total expected variance in our predicted R matrix if we know Rij,
        calculated according to our current belief about the distribution of
        Rij pretending that it's normal with mean and variance defined by our
        distribution over U and V, up to a constant common to all (i,j) pairs.
        '''
        return self._integrate_prediction(ij,
                lambda self: self.approx_pred_means_vars()[1].sum(),
                use_map=False)


    def _integrate_prediction(self, ij, fn, use_map=True):
        '''
        Calculates \int fn(new apmf) p(R_ij) dR_ij through numerical
        integration.

        If use_map, uses the current MAP estimate of R_ij; otherwise, uses
        the current approximation.
        '''
        from scipy.integrate import quad
        i, j = ij

        # which distribution for R_ij are we using?
        if use_map:
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
        scaler = math.sqrt(2 * math.pi) * std
        def calculate_fn(v):
            assert isinstance(v, float)
            apmf = deepcopy(self)
            apmf.add_rating(i, j, v)
            # apmf.fit() # not necessary, at least for now
            apmf.fit_normal()

            return fn(apmf) * np.exp(0.5 * (v - mean)**2 / var) / scaler

        # only take the expectation out to 1.96 sigma (95% of normal mass)
        left = mean - 1.96 * std
        right = mean + 1.96 * std

        est, abserr = quad(calculate_fn, left, right, epsabs=1e-2, epsrel=1e-2)
        print "(%d, %d) integrated from %g to %g:\t %g +- %g" % (
                i, j, left, right, est, abserr)
        return est


    ############################################################################
    ### Methods to actually pick a query point in active learning

    def pick_query_point(self, pool=None, key=None, procs=None):
        '''
        Use the approximation of the PMF model to select the next point to
        query, choosing elements according to key, which should be an ActivePMF
        method taking a user-item pair as its argument. Defaults to
        pred_variance.

        The choices can be limited to an iterable pool (default self.unrated).

        Spawns procs processes (default 1 per cpu), unless key.spawn_processses
        is False, procs is 1, or multiprocessing isn't available.
        '''
        if pool is None:
            pool = self.unrated
        if key is None:
            key = ActivePMF.pred_variance
        chooser = getattr(key, 'chooser', max)

        vals = self._get_key_vals(pool, key, procs)
        return chooser(zip(pool, vals), key=operator.itemgetter(1))[0]


    def _get_key_vals(self, pool, key, procs):
        try:
            if procs == 1 or not getattr(key, 'spawn_processes', True):
                raise ImportError("hackety hack hack, shouldn't propagate")
            from multiprocessing import Pool
        except ImportError:
            vals = [key(self, ij) for ij in pool]
        else:
            # TODO: use np.save instead of pickle to transfer data
            # (or maybe shared mem? http://stackoverflow.com/q/5033799/344821)
            processes = Pool(procs)
            vals = processes.map(ActivePMFEvaluator(self, key), pool)
            processes.close()
            processes.join() # make sure zombies die

        return vals

    def get_key_evals(self, pool=None, key=None, procs=None):
        '''
        Returns a matrix in the shape of the prediction matrix, with elements
        set to the value of key for each element of pool (default:
        self.unrated), and the rest of the elements set to nan.
        '''
        if pool is None:
            pool = self.unrated
        if key is None:
            key = ActivePMF.pred_variance

        evals = np.zeros((self.num_users, self.num_items))
        evals[:] = np.nan
        evals[zip(*pool)] = self._get_key_vals(pool, key, procs)
        return evals



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
    for i, j in product(xrange(apmf.num_users), xrange(apmf.num_items)):
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
    from matplotlib import cm

    pred = apmf.predicted_matrix()
    a_mean, a_var = apmf.approx_pred_means_vars()
    a_std = np.sqrt(a_var)

    xs = (real, pred, a_mean)
    norm = plt.Normalize(min(a.min() for a in xs), max(a.max() for a in xs))

    rated_alphas = zip(*((i, j, -1) for i, j in apmf.rated))
    def show_with_alpha(mat, title, subplot, alpha=.5, norm_=norm):
        plt.subplot(subplot)

        sm = cm.ScalarMappable(norm=norm_, cmap=cm.winter)
        rgba = sm.to_rgba(mat)
        rgba[rated_alphas] = alpha
        plt.imshow(rgba, norm=norm_, cmap=cm.winter, interpolation='nearest')

        plt.title(title)
        plt.colorbar()

    show_with_alpha(real, "Real", 221)
    show_with_alpha(pred, "MAP", 222)
    show_with_alpha(a_mean, "Normal: Mean", 223)
    show_with_alpha(a_std, "Normal: Std Dev", 224, 1,
            plt.Normalize(a_std.min(), a_std.max()))


def plot_criteria(apmf, keys, procs=None):
    from matplotlib import pyplot as plt
    from matplotlib import cm

    n = len(keys)
    if n <= 3:
        nr = 1
        nc = n
    else:
        nc = math.ceil(math.sqrt(n))
        nr = math.ceil(n / nc)

    for idx, key in enumerate(keys):
        evals = apmf.get_key_evals(key=key, procs=procs)

        plt.subplot(nr, nc, idx+1)
        plt.imshow(evals, interpolation='nearest', cmap=cm.winter)
        plt.title(key.nice_name)
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


KEY_FUNCS = {
    "random": ActivePMF.random_weighting,
    "pred-variance": ActivePMF.pred_variance,

    "total-variance": ActivePMF.exp_total_variance,
    "total-variance-approx": ActivePMF.exp_total_variance_byapprox,

    "uv-entropy": ActivePMF.exp_approx_entropy,
    "uv-entropy-approx": ActivePMF.exp_approx_entropy_byapprox,

    "pred-entropy-bound": ActivePMF.exp_pred_entropy_bound,
    "pred-entropy-bound-approx": ActivePMF.exp_pred_entropy_bound_byapprox,
}

def compare(key_names, plot=True, saveplot=None, latent_d=5,
            processes=None, **kwargs):
    real, ratings = make_fake_data(**kwargs)
    apmf = ActivePMF(ratings, latent_d=latent_d)

    results = []
    for key_name in key_names:
        key = KEY_FUNCS[key_name]

        print '=' * 80
        print "Starting", key.nice_name
        print '=' * 80
        results.append(list(full_test(
            deepcopy(apmf),
            real,
            key,
            key.do_normal_fit,
            processes=processes if key.spawn_processes else 1)))
        print '=' * 80
        print '=' * 80

    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.xlabel("# of rated elements")
        plt.ylabel("RMSE")

        for key_name, result in zip(key_names, results):
            plt.plot(*zip(*result), label=KEY_FUNCS[key_name].nice_name)

        plt.legend()
        if saveplot is None:
            plt.show()
        else:
            plt.savefig(saveplot)


def main():
    key_names = set(KEY_FUNCS.keys())

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

    # if we're not running interactively, use Agg
    if args.outfile:
        import matplotlib
        matplotlib.use('Agg')

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
