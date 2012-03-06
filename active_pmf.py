#!/usr/bin/env python
'''
Code to do active learning on a PMF model.
'''

from __future__ import division

from copy import deepcopy
import itertools as itools
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

class ActivePMF(object):
    def __init__(self, rating_tuples, latent_d=1):
        # the actual PMF model
        self.pmf = ProbabilisticMatrixFactorization(rating_tuples, latent_d)

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
        self.learning_rate = 1e-4
        self.min_eig = 1e-5 # minimum eigenvalue to be considered positive-def


    # easy access to relevant PMF attributes
    # TODO: just do inheritance...
    ratings = property(lambda self: self.pmf.ratings)
    rated = property(lambda self: self.pmf.rated)
    unrated = property(lambda self: self.pmf.unrated)

    sigma_sq = property(lambda self: self.pmf.sigma_sq)
    sigma_u_sq = property(lambda self: self.pmf.sigma_u_sq)
    sigma_v_sq = property(lambda self: self.pmf.sigma_v_sq)

    latent_d = property(lambda self: self.pmf.latent_d)
    num_users = property(lambda self: self.pmf.num_users)
    num_items = property(lambda self: self.pmf.num_items)

    add_rating = property(lambda self: self.pmf.add_rating)
    add_ratings = property(lambda self: self.pmf.add_ratings)

    predicted_matrix = property(lambda self: self.pmf.predicted_matrix)

    def train_pmf(self):
        '''Train the underlying PMF model to convergence.'''
        self.pmf.fit()


    def initialize_approx(self):
        '''
        Also sets up the normal approximation to be near the PMF result
        (throwing away any old approximation).
        '''
        # set mean to PMF's MAP values
        self.mean = np.hstack((self.pmf.users.reshape(-1),
                               self.pmf.items.reshape(-1)))

        # set covariance to a random positive-definite matrix
        s = np.random.normal(0, 2, (self.approx_dim, self.approx_dim))
        self.cov = project_psd(s, min_eig=self.min_eig)


    def mean_meandiff(self):
        p = np.hstack((self.pmf.users.reshape(-1), self.pmf.items.reshape(-1)))
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


    # NOTE: PMF supports weighted ratings, this doesn't
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

                for i, j, rating, weight in self.ratings)
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


    def gradient(self, mean=None, cov=None):
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
        # of the covariance matrix, but we make grad_cov symmetric anyway
        grad_mean = np.zeros_like(mean)
        grad_cov = np.zeros_like(cov)

        # TODO: vectorize as much as possible
        for i, j, rating, weight in self.ratings:
            for k in xrange(self.latent_d):
                uki = u[k, i]
                vkj = v[k, j]
                muki = mean[uki]
                mvkj = mean[vkj]

                # gradient of - E[ U_ki V_kj U_li V_lj ] / sigma^2
                for l in xrange(k+1, self.latent_d):
                    args = (uki, vkj, u[l,i], v[l,j])

                    for i, a in enumerate(args):
                        rest = args[:i] + args[i+1:]
                        grad_mean[a] += tripexpect(mean, cov, *rest) / sig

                    for a, b in itools.combinations(args, 2):
                        oth = set(args).difference([a,b])
                        c = oth.pop()
                        d = oth.pop()
                        inc = (mean[c] * mean[d] + cov[c, d]) / sig
                        grad_cov[a, b] += inc
                        grad_cov[b, a] += inc

                # gradient of - E[ U_ki^2 V_kj^2 ] / (2 sigma^2)
                grad_mean[uki] += (4 * mvkj * cov[uki,vkj]
                             + 2 * muki * (mvkj**2 + cov[vkj,vkj])) / (2*sig)
                grad_mean[vkj] += (4 * muki * cov[uki,vkj]
                             + 2 * mvkj * (mvkj**2 + cov[uki,uki])) / (2*sig)

                inc = 2 * (muki * mvkj + cov[uki,vkj]) / sig
                grad_cov[uki,vkj] += inc
                grad_cov[vkj,uki] += inc

                grad_cov[uki,uki] += (mvkj**2 + cov[vkj,vkj]) / (2*sig)
                grad_cov[vkj,vkj] += (muki**2 + cov[uki,uki]) / (2*sig)

                # gradient of Rij E[U_ki V_kj] / sigma^2
                grad_mean[uki] -= mvkj * rating / sig
                grad_mean[vkj] -= muki * rating / sig

                grad_cov[uki,vkj] -= 1
                grad_cov[vkj,uki] -= 1

        # gradient of - E[U_ki^2] / (2 sigma_u^2), same for V
        grad_mean[us] += mean[us] / self.sigma_u_sq
        grad_mean[vs] += mean[vs] / self.sigma_v_sq

        grad_cov[us,us] -= 1 # adds to diagonals only
        grad_cov[vs,vs] -= 1

        # gradient of -ln(|cov|)/2
        # need each cofactor of the matrix divided by its determinant;
        # this is just the transpose of its inverse
        # (see http://stackoverflow.com/a/6528024/344821)

        # XXX: assuming cov is invertible. this is true because we're
        # projecting, but wouldn't necessarily be true with e.g. barrier method

        grad_cov += np.linalg.inv(cov).T / 2


        return grad_mean, grad_cov


    def try_updates(self, grad_mean, grad_cov, lr):
        '''
        Returns the parameters used by taking a step in the direction of
        the passed gradient.
        '''
        new_mean = self.mean - lr * grad_mean
        new_cov = project_psd(self.cov - lr * grad_cov, min_eig=self.min_eig)
        return new_mean, new_cov


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
        lr = self.learning_rate
        old_kl = self.kl_divergence()
        converged = False

        while not converged:
            grad_mean, grad_cov = self.gradient()

            # take one step, trying different learning rates if necessary
            while True:
                #print "  setting learning rate =", lr
                new_mean, new_cov = self.try_updates(grad_mean, grad_cov, lr)
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


    def pred_variance(self, i, j):
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


    def pick_query_point(self, pool=None):
        '''
        Use the approximation of the PMF model to select the next point to
        query, based on the element of the matrix with the highest variance
        under the approximation.
        '''
        if pool is None:
            pool = self.unrated
        return max(pool, key=lambda (i,j): self.pred_variance(i, j))


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
        mask[random.randrange(num_items), zero_col] = 1

    for zero_row in np.logical_not(mask.sum(axis=1)).nonzero()[0]:
        mask[zero_row, random.randrange(num_users)] = 1

    assert (mask.sum(axis=0) > 0).all()
    assert (mask.sum(axis=1) > 0).all()

    # convert into the list-of-ratings form we want
    rates = np.zeros((mask.sum(), 4))
    for idx, (i, j) in enumerate(np.transpose(mask.nonzero())):
        rates[idx] = [i, j, ratings[i, j], 1]

    return ratings, rates


def plot_variances(apmf, vmax=None):
    from pmf import plt
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


def full_test(apmf, true, picker=ActivePMF.pick_query_point, fit_normal=True):
    print "Training PMF:"
    for ll in apmf.pmf.fit_lls():
        print "\tLL: %g" % ll

    apmf.initialize_approx()

    if fit_normal:
        print "Fitting normal:"
        for kl in apmf.fit_normal_kls():
            print "\tKL: %g" % kl

        print "Mean diff of means: %g; mean cov %g" % (
                apmf.mean_meandiff(), np.abs(apmf.cov.mean()))

    rmse = np.sqrt(((true - apmf.predicted_matrix())**2).sum())
    print "RMSE: %g" % rmse
    yield len(apmf.rated), rmse

    total = apmf.num_users * apmf.num_items

    while apmf.unrated:
        print
        #print '=' * 80
        i, j = picker(apmf)
        apmf.add_rating(i, j, true[i, j])
        print "Queried (%d, %d); %d/%d known" % (i, j, len(apmf.rated), total)

        print "Training PMF:"
        for ll in apmf.pmf.fit_lls():
            print "\tLL: %g" % ll

        if fit_normal:
            print "Fitting normal:"
            for kl in apmf.fit_normal_kls():
                print "\tKL: %g" % kl
                assert kl > 0

            print "Mean diff of means: %g; mean cov %g" % (
                    apmf.mean_meandiff(), np.abs(apmf.cov.mean()))

        rmse = np.sqrt(((true - apmf.predicted_matrix())**2).sum())
        print "RMSE: %g" % rmse
        yield len(apmf.rated), rmse


def compare(plot=True, saveplot=None, latent_d=5, **kwargs):
    true, ratings = make_fake_data(**kwargs)
    apmf = ActivePMF(ratings, latent_d=latent_d)

    uncertainty_sampling = list(full_test(deepcopy(apmf), true))
    print
    print '=' * 80
    print '=' * 80
    print '=' * 80

    pick_rand = lambda apmf: random.choice(list(apmf.unrated))
    random_sampling = list(full_test(deepcopy(apmf), true, pick_rand, False))

    if plot:
        from pmf import plt
        plt.figure()
        plt.xlabel("# of rated elements")
        plt.ylabel("RMSE")

        plt.plot(*zip(*uncertainty_sampling), label="Uncertainty Sampling")
        plt.plot(*zip(*random_sampling), label="Random")

        plt.legend()
        if saveplot is None:
            plt.show()
        else:
            plt.savefig(saveplot)

def main():
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
    parser.add_argument('outfile', nargs='?', default=None)
    args = parser.parse_args()

    try:
        compare(num_users=args.num_users, num_items=args.num_items,
                rank=args.gen_rank, latent_d=args.latent_d,
                noise=args.noise,
                rating_prob=args.rating_prob,
                plot=args.plot, saveplot=args.outfile)
    except:
        import pdb; pdb.post_mortem()

if __name__ == '__main__':
    main()
