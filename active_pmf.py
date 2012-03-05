#!/usr/bin/env python
from __future__ import division

import itertools as itools

import numpy as np

from pmf import ProbabilisticMatrixFactorization


def tripexpect(mean, cov, a, b, c):
    '''E[X_a X_b X_c] for N(mean, cov)'''
    return mean[a] * mean[b] * mean[c] + \
            mean[a]*cov[b,c] + mean[b]*cov[a,c] + mean[c]*cov[a,b]

def quadexpect(mean, cov, a, b, c, d):
    '''E[X_a X_b X_c X_d] for N(mean, cov)'''
    abcd = set((a, b, c, d))
    if len(abcd) != 4:
        raise ValueError("quadexpect only works for distinct indices")

    # product of the means
    e = mean[a] * mean[b] * mean[c] * mean[d]

    # pairs of means times cov of the other two
    for w, x in itools.combinations(abcd, 2):
        y, z = abcd.difference([w, x])
        e += mean[w] * mean[x] * cov[y, z]

    # pairs of covs
    e += cov[a,b]*cov[c,d] + cov[a,c]*cov[b,d] + cov[a,d]*cov[b,c]

    return e

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
    ratings = property(lambda self: self.pmf.ratings)
    rated = property(lambda self: self.pmf.rated)

    sigma_sq = property(lambda self: self.pmf.sigma_sq)
    sigma_u_sq = property(lambda self: self.pmf.sigma_u_sq)
    sigma_v_sq = property(lambda self: self.pmf.sigma_v_sq)

    latent_d = property(lambda self: self.pmf.latent_d)
    num_users = property(lambda self: self.pmf.num_users)
    num_items = property(lambda self: self.pmf.num_items)

    add_rating = property(lambda self: self.pmf.add_rating)
    add_ratings = property(lambda self: self.pmf.add_ratings)

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

        div = 0

        # terms based on the squared error
        sqerr = 0
        for i, j, rating, weight in self.ratings:
            sqerr += rating**2

            for k in xrange(self.latent_d):
                uki = u[k, i]
                vkj = v[k, j]
                muki = mean[uki]
                mvkj = mean[vkj]

                for l in xrange(self.latent_d):
                    if l == k: continue
                    sqerr += 2 * quadexpect(mean, cov, uki, vkj, u[l,i], v[l,j])
                
                # E[U_ki^2 V_kj^2]
                sqerr += 4 * muki * mvkj * cov[uki, vkj] \
                        + 2 * cov[uki, vkj]**2 \
                        + (muki**2 + cov[uki,uki]) * (mvkj**2 + cov[vkj, vkj])

                # - 2 Rij E[U_kj V_kj]
                sqerr -= 2 * rating * (muki * mvkj + cov[uki, vkj])
        div += sqerr / (2 * self.sigma_sq)

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

        # TODO: vectorize as much as possible?
        for i, j, rating, weight in self.ratings:
            for k in xrange(self.latent_d):
                uki = u[k, i]
                vkj = v[k, j]
                muki = mean[uki]
                mvkj = mean[vkj]

                # gradient of - E[ U_ki V_kj U_li V_lj ] / sigma^2
                for l in xrange(self.latent_d):
                    if l == k: continue
                    args = (uki, vkj, u[l,i], v[l,j])

                    for i, a in enumerate(args):
                        rest = args[:i] + args[i+1:]
                        grad_mean[a] += tripexpect(mean, cov, *rest) / sig

                    for a, b in itools.combinations(args, 2):
                        oth = set(args).difference([a,b])
                        c = oth.pop()
                        d = oth.pop()
                        inc = mean[c] * mean[d] + cov[c, d]
                        grad_cov[a, b] += inc / sig
                        grad_cov[b, a] += inc / sig

                # gradient of - E[ U_ki^2 V_kj^2 ] / (2 sigma^2)
                grad_mean[uki] += (4 * mvkj * cov[uki,vkj]
                             + 2 * muki * (mvkj**2 + cov[vkj,vkj])) / (2*sig)
                grad_mean[vkj] += (4 * muki * cov[uki,vkj]
                             + 2 * mvkj * (mvkj**2 + cov[uki,uki])) / (2*sig)

                inc = 4 * (muki * mvkj + cov[uki,vkj])
                grad_cov[uki,vkj] += inc / (2*sig)
                grad_cov[vkj,uki] += inc / (2*sig)

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


    def try_updates(self, grad_mean, grad_cov):
        '''
        Returns the parameters used by taking a step in the direction of
        the passed gradient.
        '''
        new_mean = self.mean - self.learning_rate * grad_mean
        new_cov = project_psd(self.cov - self.learning_rate * grad_cov,
                              min_eig=self.min_eig)
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
        # TODO: consider a log-barrier to stay PSD?
        # TODO: some kind of regularization to stay near orig params?
        # TODO: watch for divergence and restart?

        old_kl = self.kl_divergence()
        converged = False

        while not converged:
            grad_mean, grad_cov = self.gradient()

            # take one step, trying different learning rates if necessary
            while True:
                #print "  setting learning rate =", self.learning_rate
                new_mean, new_cov = self.try_updates(grad_mean, grad_cov)
                new_kl = self.kl_divergence(new_mean, new_cov)

                # TODO: configurable momentum, stopping conditions
                if new_kl < old_kl:
                    self.mean = new_mean
                    self.cov = new_cov
                    self.learning_rate *= 1.25

                    if old_kl - new_kl < .005:
                        converged = True
                    break
                else:
                    self.learning_rate *= .5

                    if self.learning_rate < 1e-10:
                        converged = True
                        break

            yield new_kl
            old_kl = new_kl


    def pred_variance(self, i, j):
        var = 0
        mean = self.mean
        cov = self.cov
        u = self.u
        v = self.v

        # E[U_ki V_kj U_li V_lj] for k != l
        for k in xrange(self.latent_d - 1):
            for l in xrange(k+1, self.latent_d):
                var += 2 * quadexpect(mean, cov, u[k,i], v[k,j], u[l,i], v[l,j])

        # E[U_ki^2 V_kj^2]
        for k in xrange(self.latent_d):
            var += exp_squared(mean, cov, u[k,i], v[k,j])

        # (sum E[U_ki V_kj])^2
        twoexp = mean[u[:,i]] * mean[v[:,j]] + cov[u[:,i], v[:,j]]
        var -= twoexp.sum() ** 2

        return var


    def pick_query_point(self, pool=None):
        '''
        Use the approximation of the PMF model to select the next point to
        query, based on the element of the matrix with the highest variance
        under the approximation.
        '''
        if pool is None:
            # default to all unobserved elements
            p = itools.product(xrange(self.num_users), xrange(self.num_items))
            pool = set(p).difference(self.rated)

        return max(pool, key=lambda (i,j): self.pred_variance(i, j))


def make_fake_data_apmf():
    from pmf import fake_ratings
    ratings, u, v = fake_ratings(num_users=10, num_items=10, num_ratings=5)

    apmf = ActivePMF(ratings, latent_d=5)
    apmf.true_u = u
    apmf.true_v = v

    print "Training PMF:"
    for ll in apmf.pmf.fit_lls():
        print "\tLL: %g" % ll
    print "Done training.\n"
    return apmf


def plot_variances(apmf, vmax=None):
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

if __name__ == '__main__':
    # import cPickle as pickle
    # try:
    #     with open('apmf_fake.pkl') as f:
    #         apmf = pickle.load(f)
    # except:
    #     apmf = make_fake_data_apmf()
    #     with open('apmf_fake.pkl', 'w') as f:
    #         pickle.dump(apmf, f)
    apmf = make_fake_data_apmf()

    print "\nFinding approximation:"
    apmf.initialize_approx()
    kls = []
    for kl in apmf.fit_normal_kls():
        kls.append(kl)
        print "KL:", kl

    print "Mean diff of means: %g" % (
            np.abs(apmf.mean - np.hstack((apmf.pmf.users.reshape(-1),
                                          apmf.pmf.items.reshape(-1))))
            .mean())
    print "Mean cov: %g" % np.abs(apmf.cov).mean()


    query_user, query_item = apmf.pick_query_point()
    print "Query point: %d, %d" % (query_user, query_item)

    from pmf import plt
    plt.figure()
    plt.plot(kls)
    plt.xlabel("Iteration")
    plt.ylabel("KL")

    plt.figure()
    var1 = plot_variances(apmf)
    maxvar1 = var1[np.isfinite(var1).nonzero()].max()
    print maxvar1

    query_rating = np.dot(apmf.true_u[query_user], apmf.true_v[query_item]) \
            + np.random.normal(scale=.25)
    apmf.add_rating(query_user, query_item, query_rating)
    print "Rating for %d, %d: %g" % (query_user, query_item, query_rating)

    print '\n'
    print "-" * 80
    print "Retraining PMF"
    apmf.train_pmf()
    print "Done retraining.\n"

    print "\nFinding approximation:"
    #apmf.initialize_approx() # TODO: reinitialize the approximation?
    print "Mean diff of means: %g" % (
            np.abs(apmf.mean - np.hstack((apmf.pmf.users.reshape(-1),
                                          apmf.pmf.items.reshape(-1))))
            .mean())
    print "Mean cov: %g" % np.abs(apmf.cov).mean()
    newkls = [apmf.kl_divergence()]
    print "KL:", newkls[0]
    for kl in apmf.fit_normal_kls():
        newkls.append(kl)
        print "KL:", kl

    print "Mean diff of means: %g" % (
            np.abs(apmf.mean - np.hstack((apmf.pmf.users.reshape(-1),
                                          apmf.pmf.items.reshape(-1))))
            .mean())
    print "Mean cov: %g" % np.abs(apmf.cov).mean()

    plt.figure()
    plot_variances(apmf, maxvar1)

    plt.figure()
    plt.plot(newkls)
    plt.xlabel("Retraining Iteration")
    plt.ylabel("KL")

    plt.show()
