#!/usr/bin/env python3
'''
Code to do active learning on a PMF model.
'''

from copy import deepcopy
import functools
from itertools import product, cycle
import math
import numbers
import operator
import random
import warnings

import numpy as np
from scipy import stats

from pmf import ProbabilisticMatrixFactorization

try:
    from normal_exps_cy import (quadexpect, exp_a2bc, exp_dotprod_sq,
                                normal_gradient)
except ImportError:
    warnings.warn("cython version not available; using pure-python version")
    from normal_exps import (quadexpect, exp_a2bc, exp_dotprod_sq,
                             normal_gradient)

################################################################################
### Helpers

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


# avoid pickling methods
class ActivePMFEvaluator(object):
    def __init__(self, apmf, key):
        self.apmf = apmf
        self.key_name = key.__name__

    def __call__(self, ij):
        fn = getattr(self.apmf, self.key_name)
        try:
            return fn(ij)
        except Exception:
            # show what the error was if it's in a pool
            import traceback
            traceback.print_exc()
            raise

def strictmap(*args, **kwargs):
    return list(map(*args, **kwargs))

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
    def __init__(self, rating_tuples, latent_d=1,
                 rating_values=None, discrete_expectations=False):
        # the actual PMF model
        super(ActivePMF, self).__init__(rating_tuples, latent_d)

        # make sure that the ratings matrix is in floats
        # because the cython code currently only handles floats
        self.ratings = np.array(self.ratings, dtype=float, copy=False)

        # rating values
        if rating_values:
            rating_values = set(map(float, rating_values))
            if not rating_values.issuperset(self.ratings[:,2]):
                raise ValueError("got ratings not in rating_values")

        self.rating_values = rating_values
        self.discrete_expectations = discrete_expectations

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


    rating_values = property(lambda self: self._rating_values)
    rating_bounds = property(lambda self: self._rating_bounds)

    @rating_values.setter
    def rating_values(self, vals):
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

        e_dot_sq = functools.partial(exp_dotprod_sq, u, v, mean, cov)

        # terms based on the squared error
        div = (
            sum(
                # E[ (U_i^T V_j)^2 ]
                e_dot_sq(i, j)

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
            grad_mean, grad_cov = normal_gradient(self)

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

        e_dot_sq = functools.partial(exp_dotprod_sq, self.u, self.v, mean, cov)

        # TODO: vectorize
        for i in range(self.num_users):
            us = self.u[:, i]
            for j in range(self.num_items):
                vs = self.v[:, j]
                p_mean[i, j] = m = (mean[us] * mean[vs] + cov[us, vs]).sum()
                p_var[i, j] = e_dot_sq(i, j) - m**2

        return p_mean, p_var

    def approx_pred_covs(self):
        '''
        Returns the covariance between elements of the predicted R matrix
        under the normal approximation.
        '''
        n = self.num_users
        m = self.num_items
        D = self.latent_d

        pred_covs = np.zeros((n*m, n*m))
        # covariance of U_i.V_j with U_a.V_b

        mean = self.mean
        cov = self.cov

        qexp = functools.partial(quadexpect, mean, cov)
        a2bc = functools.partial(exp_a2bc, mean, cov)
        e_dot_sq = functools.partial(exp_dotprod_sq, self.u, self.v, mean, cov)

        # TODO: vectorize, maybe cythonize
        ijs = list(product(range(n), range(m)))

        for idx1, (i, j) in enumerate(ijs):
            u_i = self.u[:,i]
            v_j = self.v[:,j]

            # variance of U_i . V_j
            pred_covs[idx1, idx1] = (
                e_dot_sq(i, j)
                - (mean[u_i] * mean[v_j] + cov[u_i, v_j]).sum() ** 2
            )

            # loop over lower triangle of the cov matrix
            for idx2 in range(idx1 + 1, len(ijs)):
                a, b = ijs[idx2]
                u_a = self.u[:, a]
                v_b = self.v[:, b]

                cv = 0

                # sum_k sum_{l != k} E[Uki Vkj Ula Vlb]
                for k in range(D):
                    # sum over l != k
                    for l in range(k):
                        cv += qexp(u_i[k], v_j[k], u_a[l], v_b[l])
                    for l in range(k+1, D):
                        cv += qexp(u_i[k], v_j[k], u_a[l], v_b[l])

                # sum_k E[Uki Uka Vkj Vkb]
                if i == a: # j != b
                    for k in range(D):
                        cv += a2bc(u_i[k], v_j[k], v_b[k])
                elif j == b: # i != a
                    for k in range(D):
                        cv += a2bc(v_j[k], u_i[k], u_a[k])
                else: # i != a, j != b
                    for k in range(D):
                        cv += qexp(u_i[k], v_j[k], u_a[k], v_b[k])

                # - sum_{k,l} E[Uki Vkj] E[Uli Vlb]
                cv -= ((mean[u_i] * mean[v_j] + cov[u_i, v_j]).sum() *
                       (mean[u_a] * mean[v_b] + cov[u_a, v_b]).sum())

                pred_covs[idx1, idx2] = cv
                pred_covs[idx2, idx1] = cv

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
    @nice_name("Pred Variance")
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
            exp_dotprod_sq(self.u, self.v, mean, cov, i, j)

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
    @nice_name("E[U/V Entropy] (MAP)")
    @minimize
    def exp_approx_entropy(self, ij):
        '''
        The expected entropy in our approximation of U and V if we know Rij,
        calculated according to our current MAP belief about the distribution
        of Rij, up to a constant common to all (i,j) pairs.
        '''
        return self._exp_with_rij(ij, ActivePMF._approx_entropy,
                use_map=True)

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("E[U/V Entropy] (Approx)")
    @minimize
    def exp_approx_entropy_byapprox(self, ij):
        '''
        The expected entropy in our approximation of U and V if we know Rij,
        calculated according to our current belief about the distribution of
        Rij pretending that it's normal with mean and variance defined by our
        distribution over U and V, up to a constant common to all (i,j) pairs.
        '''
        return self._exp_with_rij(ij, ActivePMF._approx_entropy,
                use_map=False)


    def _pred_entropy_bound(self):
        '''
        Upper bound on the entropy of the predicted matrix, up to an additive
        constant.
        '''
        p_cov = self.approx_pred_covs()
        s, logdet = np.linalg.slogdet(p_cov)
        if s != 1:
            if s == -1 and logdet < -50:
                # numerical error, pretend it's basically 0
                # TODO: track these in a way that's not as slow as printing
                return -1000 # XXX if we did det, could be 0 here
            else:
                m = "prediction cov has det with sign {}, log {}"
                raise ValueError(m.format(s, logdet))
        return logdet

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("E[Pred Entropy Bound] (MAP)")
    @minimize
    def exp_pred_entropy_bound(self, ij):
        '''
        The expectation on an upper bound of the entropy in our predicted R
        matrix if we know Rij, calculated according to our current MAP belief
        about the distribution of Rij, up to a constant common to all (i,j)
        pairs.
        '''
        return self._exp_with_rij(ij, ActivePMF._pred_entropy_bound,
                use_map=True)

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("E[Pred Entropy Bound] (Approx)")
    @minimize
    def exp_pred_entropy_bound_byapprox(self, ij):
        '''
        The expectation on an upper bound of the entropy in our predicted R
        matrix if we know Rij, calculated according to our current belief about
        the distribution of Rij pretending that it's normal with mean and
        variance defined by our distribution over U and V, up to a constant
        common to all (i,j) pairs.
        '''
        return self._exp_with_rij(ij, ActivePMF._pred_entropy_bound,
                use_map=False)

    def _total_variance(self):
        return self.approx_pred_means_vars()[1].sum()

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("E[Pred Total Variance] (MAP)")
    @minimize
    def exp_total_variance(self, ij):
        '''
        The total expected variance in our predicted R matrix if we know Rij,
        calculated according to our current MAP belief about the distribution
        of Rij, up to a constant common to all (i,j) pairs.
        '''
        return self._exp_with_rij(ij, ActivePMF._total_variance,
                use_map=True)

    @do_normal_fit(True)
    @spawn_processes(True)
    @nice_name("E[Pred Total Variance] (Approx)")
    @minimize
    def exp_total_variance_byapprox(self, ij):
        '''
        The total expected variance in our predicted R matrix if we know Rij,
        calculated according to our current belief about the distribution of
        Rij pretending that it's normal with mean and variance defined by our
        distribution over U and V, up to a constant common to all (i,j) pairs.
        '''
        return self._exp_with_rij(ij, ActivePMF._total_variance,
                use_map=False)

    def _exp_with_rij(self, ij, fn, use_map=True, discretize=None):
        '''
        Calculates E[fn(apmf with R_ij)] through numerical integration.

        If use_map, uses the current MAP estimate of R_ij; otherwise, uses a
        normal distribution for R_ij with mean and variance equal to its
        variance under the current normal approximation.

        If discretize (defaulting to self.discrete_expectations) is true and
        self.rating_values is set, compute the expectation by evaluating each
        of the rating values and summing.
        '''
        i, j = ij

        if discretize is None:
            discretize = self.discrete_expectations

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
            var = exp_dotprod_sq(self.u, self.v, self.mean, self.cov, i, j) \
                    - mean**2

        std = math.sqrt(var)

        def calculate_fn(v):
            apmf = deepcopy(self)
            apmf.add_rating(i, j, v)
            # apmf.fit() # not necessary, at least for now
            apmf.fit_normal()

            return fn(apmf)

        points = self.rating_values
        if discretize and points:
            probs = np.diff(stats.norm.cdf(self.rating_bounds,
                                           loc=mean, scale=std))
            est = sum(calculate_fn(v) * p for v, p in zip(points, probs))
            s = "summed"
        else:
            if discretize and points is None:
                warnings.warn("ActivePMF has no rating_values; doing integral")

            # only take the expectation out to 2 sigma (>95% of normal mass)
            left = mean - 2 * std
            right = mean + 2 * std

            est = stats.norm.expect(calculate_fn, loc=mean, scale=std,
                    lb=left, ub=right, epsrel=.02)
            s = "from {:5.1f} to {:5.1f}".format(left, right)

        name = getattr(fn, '__name__', '')
        print("\t{:>20}({},{}) {}: {: 10.2f}".format(name, i, j, s, est))
        return est


    ############################################################################
    ### Methods to actually pick a query point in active learning

    def pick_query_point(self, pool=None, key=None, procs=None, worker_pool=None):
        '''
        Use the approximation of the PMF model to select the next point to
        query, choosing elements according to key, which should be an ActivePMF
        method taking a user-item pair as its argument. Defaults to
        pred_variance.

        The choices can be limited to an iterable pool (default self.unrated).

        Spawns procs processes (default 1 per cpu), unless key.spawn_processses
        is False, procs is 1, or multiprocessing isn't available.

        If worker_pool is passed, ignore procs and use that pool. (If
        key.spawn_processes is False, still uses one process in that pool
        to do all the work.)
        '''
        if pool is None:
            pool = self.unrated
        if key is None:
            key = ActivePMF.pred_variance
        chooser = getattr(key, 'chooser', max)

        if len(pool) == 0:
            raise ValueError("can't pick a query point from an empty pool")
        elif len(pool) == 1:
            return next(iter(pool))

        vals = self._get_key_vals(pool, key, procs, worker_pool)
        return chooser(zip(pool, vals), key=operator.itemgetter(1))[0]


    def _get_key_vals(self, pool, key, procs, worker_pool):
        # TODO: use np.save instead of pickle to transfer data
        # (or maybe shared mem? http://stackoverflow.com/q/5033799/344821)

        evaluator = ActivePMFEvaluator(self, key)

        if procs == 1 or not getattr(key, 'spawn_processes', True):
            if worker_pool is not None:
                if not hasattr(worker_pool, 'access_lock'):
                    return worker_pool.apply(strictmap, (evaluator, pool))
                else:
                    with worker_pool.access_lock:
                        result = worker_pool.apply_async(
                                                strictmap, (evaluator, pool))
                    return result.get()
            else:
                return [key(self, ij) for ij in pool]
        else:
            if worker_pool is not None:
                if not hasattr(worker_pool, 'access_lock'):
                    return worker_pool.map(evaluator, pool)
                else:
                    with worker_pool.access_lock:
                        result = worker_pool.map_async(evaluator, pool)
                    return result.get()
            else:
                from multiprocessing import Pool
                worker_pool = Pool(procs)
                vals = worker_pool.map(evaluator, pool)
                worker_pool.close()
                worker_pool.join()
                return vals


    def get_key_evals(self, pool=None, key=None, procs=None, worker_pool=None):
        '''
        Returns a matrix in the shape of the prediction matrix, with elements
        set to the value of key for each element of pool (default:
        self.unrated), and the rest of the elements set to nan.
        '''
        if pool is None:
            pool = self.unrated
        if key is None:
            key = ActivePMF.pred_variance

        evals = np.empty((self.num_users, self.num_items))
        evals.fill(np.nan)
        evals[list(zip(*pool))] = \
                self._get_key_vals(pool, key, procs, worker_pool)
        return evals



################################################################################
### Plotting code

def plot_variances(apmf, vmax=None):
    from matplotlib import pyplot as plt
    var = np.zeros((apmf.num_users, apmf.num_items))
    total = 0
    for i, j in product(range(apmf.num_users), range(apmf.num_items)):
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

    rated_alphas = list(zip(*((i, j, -1) for i, j in apmf.rated)))
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


def plot_rmses(results):
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    plt.xlabel("# of rated elements")
    plt.ylabel("RMSE")

    # cycle through colors and line styles
    colors = 'bgrcmyk'
    linestyles = ['-', '--', ':']
    l_c = cycle(product(linestyles, colors))

    # offset lines a bit so you can see when some of them overlap
    total = len(results)
    offset = .15 / total

    nice_results = ((KEY_FUNCS[k].nice_name, k, v)
                    for k, v in results.items() if not k.startswith('_'))

    for idx, (nice_name, key_name, result) in enumerate(sorted(nice_results)):
        nums, rmses, ijs, vals = zip(*result)

        nums = np.array(nums) + (idx - total/2) * offset

        l, c = next(l_c)
        plt.plot(nums, rmses, linestyle=l, color=c, label=nice_name, marker='^')

    # only show integer values for x ticks
    xmin, xmax = plt.xlim()
    plt.xticks(range(math.ceil(xmin), math.floor(xmax) + 1))

    plt.legend(loc='best', prop=FontProperties(size=10))


def subplot_config(n):
    if n <= 3:
        return 1, n
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)
    return nr, nc


def plot_criteria(apmf, keys, procs=None):
    from matplotlib import pyplot as plt
    from matplotlib import cm

    nr, nc = subplot_config(len(keys))

    for idx, key in enumerate(keys):
        evals = apmf.get_key_evals(key=key, procs=procs)

        plt.subplot(nr, nc, idx+1)
        plt.imshow(evals, interpolation='nearest', cmap=cm.winter)
        plt.title(key.nice_name)
        plt.colorbar()


def plot_criteria_over_time(name, result, cmap=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    if cmap is None:
        from matplotlib import cm
        cmap = cm.jet

    nums, rmses, ijs, valses = zip(*result)

    assert ijs[0] is None
    assert valses[0] is None
    ijs = ijs[1:]
    valses = valses[1:]

    if np.all(np.isnan(valses[-1])):
        ijs = ijs[:-1]
        valses = valses[:-1]

    nr, nc = subplot_config(len(ijs))

    fig = plt.figure()
    fig.suptitle(name)
    grid = ImageGrid(fig, 111, nrows_ncols=(nr,nc), axes_pad=.3,
            cbar_location='right', cbar_mode='single')

    n_users, n_items = valses[0].shape
    xticks = np.linspace(-.5, n_items - .5, n_items + 1)
    yticks = np.linspace(-.5, n_users - .5, n_users + 1)

    vmin = min(vals[np.isfinite(vals)].min() for vals in valses)
    vmax = max(vals[np.isfinite(vals)].max() for vals in valses)
    norm = plt.Normalize(vmin, vmax)
    # TODO: dynamically adjust color range to be more distinguishable?

    for idx, (n, rmse, (i,j), vals) in enumerate(zip(nums, rmses, ijs, valses)):
        # we know n values and have RMSE of rmse, then pick ij based on vals

        grid[idx].set_title("{}: ({:.3})".format(n + 1, rmse))

        im = grid[idx].imshow(vals, interpolation='nearest', cmap=cmap,
                   origin='upper', aspect='equal', norm=norm)

        grid[idx].set_xticks(xticks)
        grid[idx].set_yticks(yticks)
        grid[idx].set_xticklabels([])
        grid[idx].set_yticklabels([])
        grid[idx].set_xlim(xticks[0], xticks[-1])
        grid[idx].set_ylim(yticks[0], yticks[-1])
        grid[idx].grid()

        # mark the selected point (indices are transposed)
        grid[idx].scatter(j, i, marker='s', c='white', s=20)

    for idx in range(len(ijs), nr * nc):
        grid[idx].set_visible(False)

    grid.cbar_axes[0].colorbar(im)

    return fig


def plot_all_criteria(results, filename_format):
    from matplotlib import pyplot as plt

    for name, result in results.items():
        nice_name = KEY_FUNCS[name].nice_name
        filename = filename_format.format(name)
        print("doing", filename)

        fig = plt.figure()
        plot_criteria_over_time(nice_name, result)
        fig.savefig(filename)
        fig.close()


################################################################################
### Testing code

def full_test(apmf, real, picker_key=ActivePMF.pred_variance,
              fit_normal=True, processes=None):
    print("Training PMF")
    for ll in apmf.fit_lls():
        pass #print "\tLL: %g" % ll

    apmf.initialize_approx()

    if fit_normal:
        print("Fitting normal")
        for kl in apmf.fit_normal_kls():
            pass #print "\tKL: %g" % kl
            assert kl > -1e5

        print("Mean diff of means: %g; mean cov %g" % (
                apmf.mean_meandiff(), np.abs(apmf.cov.mean())))

    total = apmf.num_users * apmf.num_items
    rmse = apmf.rmse(real)
    print("RMSE: {:.5}".format(rmse))
    yield len(apmf.rated), rmse, None, None


    while apmf.unrated:
        print()
        #print '=' * 80

        print("Picking a query point...")
        if len(apmf.unrated) == 1:
            i, j = next(iter(apmf.unrated))
        else:
            vals = apmf.get_key_evals(key=picker_key, procs=processes)
            i, j = picker_key.chooser(zip(apmf.unrated, vals),
                                      key=operator.itemgetter(1))[0]

        apmf.add_rating(i, j, real[i, j])
        print("Queried (%d, %d); %d/%d known" % (i, j, len(apmf.rated), total))

        print("Training PMF")
        for ll in apmf.fit_lls():
            pass # print "\tLL: %g" % ll

        if fit_normal:
            print("Fitting normal")
            for kl in apmf.fit_normal_kls():
                pass # print "\tKL: %g" % kl
                assert kl > -1e5

            print("Mean diff of means: %g; mean cov %g" % (
                    apmf.mean_meandiff(), np.abs(apmf.cov.mean())))

        rmse = apmf.rmse(real)
        print("RMSE: {:.5}".format(rmse))
        yield len(apmf.rated), rmse, (i,j), vals


def _in_between_work(apmf, i, j, realval, total, fit_normal, name):
    apmf.add_rating(i, j, realval)
    print("{:<40} Queried ({}, {}); {}/{} known".format(
            name, i, j, len(apmf.rated), total))

    apmf.fit()
    if fit_normal:
        apmf.fit_normal()

    return apmf


def _full_test_threaded(apmf, real, picker_key, fit_normal, worker_pool):
    total = real.size
    name = picker_key.nice_name

    rmse = apmf.rmse(real)
    print("{:<40} Initial RMSE: {:.5}".format(name, rmse))
    yield len(apmf.rated), rmse, None, None

    while apmf.unrated:
        n = len(apmf.rated) + 1

        print("{:<40} Picking query point {}...".format(name, n))
        if len(apmf.unrated) == 1:
            vals = np.empty((apmf.num_users, apmf.num_items))
            vals.fill(np.nan)
            i, j = next(iter(apmf.unrated))
        else:
            vals = apmf.get_key_evals(key=picker_key, worker_pool=worker_pool)
            i, j = picker_key.chooser(apmf.unrated, key=vals.__getitem__)

        apmf = worker_pool.apply(_in_between_work,
                (apmf, i, j, real[i,j], total, fit_normal, name))

        rmse = apmf.rmse(real)
        print("{:<40} RMSE {}: {:.5}".format(picker_key.nice_name, n, rmse))
        yield len(apmf.rated), rmse, (i,j), vals



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


def make_fake_data(noise=.25, num_users=10, num_items=10,
                   mask_type=0, data_type='float', rank=5,
                   u_mean=0, u_std=2, v_mean=0, v_std=2):
    # generate the true fake data
    u = np.random.normal(u_mean, u_std, (num_users, rank))
    v = np.random.normal(v_mean, v_std, (num_items, rank))

    ratings = np.dot(u, v.T)
    if noise:
        ratings += np.random.normal(0, noise, (num_users, num_items))

    # TODO: better options for data_type
    if data_type == 'float':
        vals = None
    elif data_type == 'int':
        ratings = np.round(ratings).astype(int)
        vals = None # TODO: support integrating over all integers?
    elif data_type == 'binary':
        ratings = (ratings > .5).astype(int)
        vals = {0, 1}
    elif isinstance(data_type, numbers.Integral):
        ratings = np.minimum(np.maximum(np.round(ratings), 0),
                             data_type).astype(int)
        vals = range(data_type + 1)
    else:
        raise ValueError("Don't know how to interpret data_type '{}'".format(
            data_type))

    # make the mask deciding on which things are rated
    if isinstance(mask_type, numbers.Real):
        mask = np.random.binomial(1, mask_type, ratings.shape)

    elif mask_type in {'diag', 'diagonal'}:
        mask = np.zeros_like(ratings)
        np.fill_diagonal(mask, 1)

    elif mask_type in {'diag-plus'}:
        mask = np.zeros_like(ratings)
        np.fill_diagonal(mask, 1)

        if num_users != num_items:
            warnings.warn("diag-plus doesn't work for non-square; doing diag")
        else:
            # set the k=1 diagonal, except do (-1, 1) instead of (0, 1)
            # then all rows and columns have two entries, except first has 1
            n = num_users
            mask[-1, 1] = 1
            mask[range(1,n-1), range(2,n)] = 1

    else:
        raise ValueError("Don't know how to interpret mask_type '{}'".format(
            mask_type))

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

    return ratings, rates, vals



def compare(key_names, latent_d=5, processes=None, do_threading=True,
            discrete_exp=False, **kwargs):
    import multiprocessing as mp
    from threading import Thread, Lock

    real, ratings, rating_vals = make_fake_data(**kwargs)
    apmf = ActivePMF(ratings, latent_d=latent_d,
            rating_values=rating_vals, discrete_expectations=discrete_exp)

    results = {'_real': real, '_ratings': ratings}

    if do_threading:
        # initial fit is common to all methods
        print("Doing initial fit")
        apmf.fit()
        apmf.initialize_approx()
        if any(KEY_FUNCS[name].do_normal_fit for name in key_names):
            print("Initial approximation fit")
            apmf.fit_normal()
            print("Mean diff of means: {}; mean cov {}\n".format(
                    apmf.mean_meandiff(), np.abs(apmf.cov.mean())))

        worker_pool = mp.Pool(processes)
        worker_pool.access_lock = Lock()

        def eval_key(key_name):
            key = KEY_FUNCS[key_name]

            results[key_name] = list(_full_test_threaded(
                deepcopy(apmf),
                real,
                key,
                key.do_normal_fit,
                worker_pool=worker_pool))


        threads = [Thread(name=key_name, target=eval_key, args=(key_name,))
                   for key_name in key_names]
        for thread in threads: thread.start()
        for thread in threads: thread.join()

        worker_pool.close()
        worker_pool.join()

    else:
        for key_name in key_names:
            key = KEY_FUNCS[key_name]
            results[key_name] = list(full_test(
                    deepcopy(apmf), real, key, key.do_normal_fit, processes))

    return results


def add_bool_opt(parser, name, default=False):
    parser.add_argument('--' + name, action='store_true', default=default)
    parser.add_argument('--no-' + name, action='store_false',
            dest=name.replace('-', '_'))

def main():
    import argparse
    import os
    import pickle
    import sys

    key_names = set(KEY_FUNCS.keys())
    types = {'float', 'int', 'binary'}

    parser = argparse.ArgumentParser()

    model = parser.add_argument_group("Model Options")
    model.add_argument('--latent-d', '-D', type=int, default=5)
    add_bool_opt(model, 'discrete-integration', False)
    model.add_argument('keys', nargs='*',
            help="Choices: {}.".format(', '.join(sorted(key_names))))

    problem_def = parser.add_argument_group("Problem Definiton")
    problem_def.add_argument('--gen-rank', '-R', type=int, default=5)
    problem_def.add_argument('--type', default='float',
            help="An integer (meaning values are from 0 to that integer) or "
                 "one of {}".format(', '.join(sorted(types))))

    problem_def.add_argument('--u-mean', type=float, default=0)
    problem_def.add_argument('--u-std', type=float, default=2)
    problem_def.add_argument('--v-mean', type=float, default=0)
    problem_def.add_argument('--v-std', type=float, default=2)

    problem_def.add_argument('--noise', '-n', type=float, default=.25)
    problem_def.add_argument('--num-users', '-N', type=int, default=10)
    problem_def.add_argument('--num-items', '-M', type=int, default=10)
    problem_def.add_argument('--mask', '-m', default=0)

    running = parser.add_argument_group("Running")
    running.add_argument('--processes', '-P', type=int, default=None)
    add_bool_opt(running, 'threading', True)

    results = parser.add_argument_group("Results")
    results.add_argument('--load-results', default=None, metavar='FILE')
    results.add_argument('--save-results', default=None, nargs='?',
            metavar='FILE')
    results.add_argument('--no-save-results',
            action='store_false', dest='save_results')

    plotting = parser.add_argument_group("Plotting")
    add_bool_opt(plotting, 'plot', None)
    plotting.add_argument('--outfile', default=None, metavar='FILE')
    add_bool_opt(plotting, 'plot-criteria', False)
    plotting.add_argument('--cmap', default='Accent')
    plotting.add_argument('--criteria-file', default=None, metavar='FORMAT',
            help="A {}-style format string, where {} is the key name.")
    plotting.add_argument('--outdir', default=None, metavar='DIR')

    args = parser.parse_args()


    # set defaults that are different for loading vs running
    if args.save_results is None:
        args.save_results = not args.load_results
    if args.plot is None:
        args.plot = not args.load_results

    try:
        args.mask = float(args.mask)
    except ValueError:
        pass

    try:
        args.type = int(args.type)
    except ValueError:
        if args.type not in types:
            raise ValueError("--type must be integer or one of {}".format(
                ', '.join(sorted(types))))

    # check that args.keys are valid
    for k in args.keys:
        if k not in key_names:
            sys.stderr.write("Invalid key name %s; options are %s.\n" % (
                k, ', '.join(sorted(key_names))))
            sys.exit(1)

    # try to make the out directory if necessary, set other paths based on it
    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        if args.save_results is True:
            args.save_results = os.path.join(args.outdir, 'results.pkl')

        if not args.outfile:
            args.outfile = os.path.join(args.outdir, 'rmses.png')

        if not args.criteria_file:
            args.criteria_file = os.path.join(args.outdir, '{}.png')

    # get or load results
    if args.load_results is not None:
        with open(args.load_results, 'rb') as resultsfile:
            results = pickle.load(resultsfile)

        # check args.keys are actually in the results
        if not args.keys:
            args.keys = list(k for k in results.keys() if not k.startswith('_'))
        else:
            good_keys = []
            for k in args.keys:
                if k in results:
                    good_keys.append(k)
                else:
                    warnings.warn("WARNING: requested key {} not in the saved "
                                  "results.".format(k))
            args.keys = good_keys

    else:
        if not args.keys:
            args.keys = sorted(key_names)

        try:
            results = compare(args.keys,
                    num_users=args.num_users, num_items=args.num_items,
                    u_mean=args.u_mean, u_std=args.u_std,
                    v_mean=args.v_mean, v_std=args.v_std,
                    noise=args.noise, mask_type=args.mask,
                    rank=args.gen_rank, latent_d=args.latent_d,
                    discrete_exp=args.discrete_integration,
                    data_type=args.type,
                    processes=args.processes, do_threading=args.threading)
        except Exception:
            import traceback
            print()
            traceback.print_exc()

            import pdb
            print()
            pdb.post_mortem()

            sys.exit(1)

        results['_args'] = args

    # save the results file
    if args.save_results:
        if args.save_results is not True:
            filename = args.save_results
        elif args.outfile:
            filename = args.outfile + '.pkl'
        elif args.criteria_file:
            filename = args.criteria_file.format('results') + '.pkl'
        else:
            filename = 'results.pkl'

        print("saving results in '{}'".format(filename))

        with open(filename, 'wb') as f:
            pickle.dump(results, f)


    # do any plotting
    if args.plot or args.plot_criteria:
        interactive = ((args.plot and not args.outfile) or
                       (args.plot_criteria and not args.criteria_file))

        if not interactive:
            import matplotlib
            matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        if args.plot:
            print("Plotting RMSEs")
            fig = plt.figure()
            plot_rmses(results)

            if args.outfile:
                fig.savefig(args.outfile, bbox_inches='tight', pad_inches=.1)

        if args.plot_criteria:
            from matplotlib import cm
            cmap = cm.get_cmap(args.cmap)

            for name, result in results.items():
                if name not in args.keys:
                    continue

                nice_name = KEY_FUNCS[name].nice_name
                print("Plotting {}".format(nice_name))

                fig = plot_criteria_over_time(nice_name, result, cmap)

                if args.criteria_file:
                    fig.savefig(args.criteria_file.format(name),
                            bbox_inches='tight', pad_inches=.1)

        if interactive:
            plt.show()


if __name__ == '__main__':
    main()
