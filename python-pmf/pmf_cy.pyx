#!/usr/bin/env python3
'''
An implementation of Probabilistic Matrix Factorization:
    Salakhutdinov, R., & Mnih, A. (2007). Probabilistic Matrix Factorization.
    Advances in Neural Information Processing Systems.

Loosely based on code by Danny Tarlow:
http://blog.smellthedata.com/2009/06/netflix-prize-tribute-recommendation.html
'''

from __future__ import print_function # silly cython

import itertools
import random
import warnings
from copy import deepcopy

import numpy as np
cimport numpy as np

from libc.math cimport log

cimport cython

cpdef float rmse(np.ndarray exp, np.ndarray obs) except -1:
    return np.sqrt(np.mean((obs - exp) ** 2))

cpdef float rmse_on(np.ndarray exp, np.ndarray obs, np.ndarray on) except -1:
    return np.sqrt(np.mean((obs[on] - exp[on]) ** 2))

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef class ProbabilisticMatrixFactorization:
    def __cinit__(self):
        self.learning_rate = 1e-4
        self.min_learning_rate = 1e-10
        self.stop_thresh = 1e-2

        self.sigma_sq = 1
        self.sigma_u_sq = 10
        self.sigma_v_sq = 10

        # negative variance means not to use a prior here
        # would set them to None, but that doesn't work in cython
        self.sig_u_mean = self.sig_v_mean = 0
        self.sig_u_var = self.sig_v_var = -1

    def __init__(self, np.ndarray rating_tuples not None,
                 int latent_d=1, bint subtract_mean=False,
                 object knowable=None, tuple fit_type=('batch',)):
        self.latent_d = latent_d
        self.subtract_mean = subtract_mean

        if fit_type is None:
            warnings.warn('passed None fit_type; using batch')
            fit_type = ('batch',)
        self.fit_type = fit_type

        self.ratings = np.array(rating_tuples, dtype=float, copy=False)
        if self.ratings.shape[1] != 3:
            raise TypeError("invalid rating tuple length")
        self.mean_rating = np.mean(self.ratings[:,2])

        cdef int n, m
        self.num_users = n = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = m = int(np.max(self.ratings[:, 1]) + 1)

        self.rated = set(map(tuple, self.ratings[:,:2]))
        if knowable is None:
            knowable = itertools.product(range(n), range(m))
        self.unrated = set(knowable).difference(self.rated)

        self.users = np.random.random((n, self.latent_d))
        self.items = np.random.random((m, self.latent_d))


    cpdef ProbabilisticMatrixFactorization __copy__(self):
        cdef ProbabilisticMatrixFactorization res
        res = type(self)(self.ratings, self.latent_d)
        res.__setstate__(res.__getstate__())
        return res

    cpdef ProbabilisticMatrixFactorization __deepcopy__(self, object memodict):
        cdef ProbabilisticMatrixFactorization res
        res = type(self)(self.ratings.copy())# , self.latent_d, self.subtract_mean)
        res.__setstate__(deepcopy(self.__getstate__(), memodict))
        return res

    def __setstate__(self, dict state not None):
        for k, v in state.items():
            if k == '__dict__':
                for real_k, real_v in v.items():
                    setattr(self, real_k, real_v)
            else:
                setattr(self, k, v)

    def __getstate__(self):
        return dict(
            latent_d=self.latent_d,
            num_users=self.num_users,
            num_items=self.num_items,

            learning_rate=self.learning_rate,
            min_learning_rate=self.min_learning_rate,
            stop_thresh=self.stop_thresh,

            sigma_sq=self.sigma_sq,
            sigma_u_sq=self.sigma_u_sq,
            sigma_v_sq=self.sigma_v_sq,
            fit_type=self.fit_type,

            sig_u_mean=self.sig_u_mean,
            sig_u_var=self.sig_u_var,
            sig_v_mean=self.sig_v_mean,
            sig_v_var=self.sig_v_var,

            ratings=self.ratings,
            users=self.users,
            items=self.items,

            subtract_mean=self.subtract_mean,
            mean_rating=self.mean_rating,
            rated=self.rated,
            unrated=self.unrated,
        )

    def add_rating(self, int i, int j, DTYPE_t rating):
        self.add_ratings([i, j, rating])

    def add_ratings(self, extra):
        cdef int rows = self.ratings.shape[0]
        cdef int cols = self.ratings.shape[1]

        extra = np.array(extra, copy=False, ndmin=2)
        if len(extra.shape) != 2 or extra.shape[1] != cols:
            raise TypeError("bad shape for extra")

        assert np.max(extra[:,0] + 1) <= self.num_users
        assert np.max(extra[:,1] + 1) <= self.num_items

        rating_vals = getattr(self, 'rating_values', None)
        if rating_vals is not None:
            if not set(rating_vals).issuperset(extra[:,2]):
                raise ValueError("got ratings with bad values")

        new_items = set((int(i), int(j)) for i, j in extra[:,:2])

        if not new_items.isdisjoint(self.rated):
            raise ValueError("can't rate already rated items")
        self.rated.update(new_items)
        self.unrated.difference_update(new_items)

        self.ratings = np.append(self.ratings, extra, 0)
        self.mean_rating = np.mean(self.ratings[:,2])
        # TODO: this can be done without a copy by .resize()...

    cpdef double prediction_for(self, int i, int j,
            np.ndarray users=None, np.ndarray items=None) except? 1492:
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        if self.subtract_mean:
            return np.dot(users[i], items[j]) + self.mean_rating
        else:
            return np.dot(users[i], items[j])

    @cython.cdivision(True)
    cpdef double log_likelihood(self, np.ndarray users=None,
                                      np.ndarray items=None) except? 1492:
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        cdef int i, j
        cdef double rating, r_hat, sq_error

        # TODO: is it faster to just make the whole matrix? probably...
        # TODO: save prediction_for into a function pointer
        sq_error = 0.
        for i, j, rating in self.ratings:
            r_hat = self.prediction_for(i, j, users, items)
            sq_error += (rating - r_hat)**2

        cdef double user_norm2 = np.sum(users * users)
        cdef double item_norm2 = np.sum(items * items)

        return (- sq_error   / (2. * self.sigma_sq)
                - user_norm2 / (2. * self.sigma_u_sq)
                - item_norm2 / (2. * self.sigma_v_sq))

    cpdef double ll_prior_adjustment(self) except? 1492:
        return -.5 * (
                np.log(self.sigma_sq) * self.ratings.shape[0]
                + self.num_users * self.latent_d * np.log(self.sigma_u_sq)
                + self.num_items * self.latent_d * np.log(self.sigma_v_sq))

    cpdef double full_ll(self, users=None, items=None) except? 1492:
        return self.log_likelihood(users, items) + self.ll_prior_adjustment()

    @cython.cdivision(True)
    cpdef tuple gradient(self, np.ndarray ratings=None):
        cdef double sig = self.sigma_sq
        cdef int i, j
        cdef double rating, r_hat
        cdef np.ndarray users = self.users
        cdef np.ndarray items = self.items
        if ratings is None:
            ratings = self.ratings

        cdef np.ndarray[DTYPE_t,ndim=2] grad_u = -users / self.sigma_u_sq
        cdef np.ndarray[DTYPE_t,ndim=2] grad_v = -items / self.sigma_v_sq

        for i, j, rating in ratings: # TODO: switch ratings to a buffer?
            # TODO: break this out into a function pointer?
            r_hat = self.prediction_for(i, j, users, items)
            grad_u[i, :] += items[j, :] * ((rating - r_hat) / sig)
            grad_v[j, :] += users[i, :] * ((rating - r_hat) / sig)

        return grad_u, grad_v

    @cython.cdivision(True)
    cpdef update_sigma(self):
        cdef int i, j
        cdef double sq_error = 0, rating

        for i, j, rating in self.ratings:
            r_hat = self.prediction_for(i, j)
            sq_error += (rating - r_hat)**2

        self.sigma_sq = sq_error / self.ratings.shape[0]

    @cython.cdivision(True)
    cpdef update_sigma_uv(self):
        cdef int d = self.latent_d
        cdef int n = self.num_users
        cdef int m = self.num_items

        cdef double user_norm2 = np.sum(self.users * self.users)
        cdef double item_norm2 = np.sum(self.items * self.items)

        if self.sig_u_var > 0:
            self.sigma_u_sq = user_norm2 / (n * d + 2 +
                    2*(log(self.sigma_u_sq) - self.sig_u_mean) / self.sig_u_var)
        else:
            self.sigma_u_sq = user_norm2 / n / d

        if self.sig_v_var > 0:
            self.sigma_v_sq = item_norm2 / (m * d + 2 +
                    2*(log(self.sigma_v_sq) - self.sig_v_mean) / self.sig_v_var)
        else:
            self.sigma_v_sq = item_norm2 / m / d

    def fit_lls(self):
        cdef np.ndarray grad_u, grad_v, new_users, new_items
        cdef double lr = self.learning_rate

        cdef double old_ll = self.log_likelihood()
        cdef double new_ll

        cdef bint converged = False
        while not converged:
            grad_u, grad_v = self.gradient()

            # take one step, trying different learning rates if necessary
            while True:
                #print "  setting learning rate =", lr
                new_users = self.users + lr * grad_u
                new_items = self.items + lr * grad_v
                new_ll = self.log_likelihood(new_users, new_items)

                if new_ll > old_ll:
                    self.users = new_users
                    self.items = new_items
                    lr *= 1.25

                    if new_ll - old_ll < self.stop_thresh:
                        converged = True

                    yield new_ll
                    old_ll = new_ll
                    break
                else:
                    lr *= .5

                    if lr < self.min_learning_rate:
                        converged = True
                        break

    def fit(self):
        cdef double ll
        for ll in self.fit_lls():
            pass

    def do_fit(self):
        kind, *args = self.fit_type
        if kind == 'batch':
            self.fit(*args)
        elif kind == 'mini-valid':
            self.fit_minibatches_until_validation(*args)
        else:
            raise ValueError("unknown fit type '{}'".format(kind))


    @cython.cdivision(True)
    def fit_minibatches(self, int batch_size, float lr=1, float momentum=.8,
                              np.ndarray ratings=None):
        # NOTE: this randomly shuffles ratings / self.ratings
        if ratings is None:
            ratings = self.ratings
        cdef int num_ratings = ratings.shape[0]


        cdef np.ndarray u_inc = np.zeros((self.num_users, self.latent_d))
        cdef np.ndarray v_inc = np.zeros((self.num_items, self.latent_d))

        cdef int batch_start, batch_end, n
        cdef float err
        cdef np.ndarray batch_ratings, grad_u, grad_v, pred, train_pred

        while True:
            np.random.shuffle(ratings)

            # follow gradient on minibatches
            # if it's not evenly divisible, we'll have one smaller batch
            for batch_start from 0 <= batch_start < num_ratings by batch_size:
                batch_end = batch_start + batch_size
                if batch_end > num_ratings:
                    batch_end = num_ratings
                n = batch_end - batch_start

                batch_ratings = ratings[batch_start:batch_end, :]
                grad_u, grad_v = self.gradient(batch_ratings)

                u_inc *= momentum
                u_inc += grad_u * (lr / n)
                self.users += u_inc

                v_inc *= momentum
                v_inc += grad_v * (lr / n)
                self.items += v_inc

            # get training error
            pred = self.predicted_matrix()
            train_pred = pred[tuple(self.ratings[:, :2].astype(int).T)]
            err = np.sqrt(np.mean((train_pred - self.ratings[:, 2]) ** 2))

            yield err

    def fit_minibatches_validation(self, int batch_size, int valid_size,
                                   **kwargs):
        cdef int total = self.ratings.shape[0]
        cdef set valid_idx_set = set(random.sample(range(total), valid_size))
        cdef tuple train_idx = tuple(
                i for i in range(total) if i not in valid_idx_set)
        cdef np.ndarray train = self.ratings[train_idx,:]

        cdef list valid_idx = list(valid_idx_set)
        cdef tuple valid_ijs = tuple(self.ratings[valid_idx, :2].T.astype(int))
        cdef np.ndarray valid_real = self.ratings[valid_idx, 2]

        cdef np.ndarray valid_pred
        cdef float train_err, valid_err

        for train_err in self.fit_minibatches(batch_size, ratings=train,
                                              **kwargs):
            valid_pred = self.predicted_matrix()[valid_ijs]
            valid_err = np.sqrt(np.mean((valid_pred - valid_real) ** 2))
            yield train_err, valid_err

    def fit_minibatches_until_validation(self, *args, stop_thresh=1e-3, **kw):
        cdef float last_valid, train, valid

        last_valid = np.inf
        for train, valid in self.fit_minibatches_validation(*args, **kw):
            if valid > last_valid - stop_thresh:
                break
            last_valid = valid


    @cython.cdivision(True)
    def fit_with_sigmas_lls(self, int noise_every=5, int users_every=2):
        cdef int i
        #cdef double ll # causes segfault...
        cdef bint cont = True

        while cont:
            cont = False
            for i, ll in enumerate(self.fit_lls()):
                if i % noise_every == 0:
                    self.update_sigma()
                if i % users_every == 0:
                    self.update_sigma_uv()

                yield ll

                cont = True # continue if we made any steps

            self.update_sigma()
            self.update_sigma_uv()

    def fit_with_sigmas(self, int noise_every=10, int users_every=5):
        cdef double ll
        for ll in self.fit_with_sigmas_lls(noise_every, users_every):
            pass

    cpdef np.ndarray predicted_matrix(self,
                        np.ndarray u=None, np.ndarray v=None):
        if u is None:
            u = self.users
        if v is None:
            v = self.items

        cdef np.ndarray pred = np.dot(u, v.T)
        if self.subtract_mean:
            pred += self.mean_rating
        return pred

    cpdef double rmse(self, np.ndarray real, np.ndarray on=None) except -1:
        if on is None:
            return rmse(self.predicted_matrix(), real)
        else:
            return rmse_on(self.predicted_matrix(), real, on)

    def print_latent_vectors(self):
        cdef int i, j
        print("Users:")
        for i in range(self.num_users):
            print("%d: %s" % (i, self.users[i, :]))

        print("\nItems:")
        for j in range(self.num_items):
            print("%d: %s" % (j, self.items[j, :]))


    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent_d)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent_d)


def parse_fit_type(string):
    parts = string.split(',')
    res = []
    for x in parts:
        for fn in (int, float):
            try:
                res.append(fn(x))
                break
            except ValueError:
                pass
        else:
            res.append(x)
    return tuple(res)

################################################################################
### Testing code

def fake_ratings(noise=.25, num_users=100, num_items=100, num_ratings=30,
                 latent_dimension=10):
    # Generate the latent user and item vectors
    u = np.random.normal(0, 2, (num_users, latent_dimension))
    v = np.random.normal(0, 2, (num_items, latent_dimension))

    # Get num_ratings ratings per user.
    ratings = []
    for i in range(num_users):
        for j in random.sample(range(num_items), num_ratings):
            rating = np.dot(u[i], v[j]) + np.random.normal(scale=noise)
            ratings.append((i, j, rating))

    return (np.array(ratings), u, v)


def plot_ratings(ratings):
    import matplotlib.pyplot as plt
    plt.plot(ratings[:, 1], ratings[:, 2], 'bx')
    plt.show()


def plot_latent_vectors(U, V):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig = plt.figure()
    ax = fig.add_subplot(121)
    cmap = cm.jet
    ax.imshow(U, cmap=cmap, interpolation='nearest')
    plt.title("Users")
    plt.axis("off")

    ax = fig.add_subplot(122)
    ax.imshow(V, cmap=cmap, interpolation='nearest')
    plt.title("Items")
    plt.axis("off")


def plot_predicted_ratings(U, V):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    u_rows, u_cols = U.shape
    v_rows, v_cols = V.shape

    r_hats = -5 * np.ones((u_rows + u_cols + 1, v_rows + v_cols + 1))

    for i in range(u_rows):
        for j in range(u_cols):
            r_hats[i + v_cols + 1, j] = U[i, j]

    for i in range(v_rows):
        for j in range(v_cols):
            r_hats[j, i + u_cols + 1] = V[i, j]

    for i in range(u_rows):
        for j in range(v_rows):
            r_hats[i + u_cols + 1, j + v_cols + 1] = np.dot(U[i], V[j]) / 10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(r_hats, cmap=cm.gray, interpolation='nearest')
    plt.title("Predicted Ratings")
    plt.axis("off")


def main():
    import matplotlib.pyplot as plt

    (ratings, true_o, true_d) = fake_ratings()
    #plot_ratings(ratings)

    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=5)

    lls = []
    for ll in pmf.fit_lls():
        lls.append(ll)
        print("LL =", ll)

    plt.figure()
    plt.plot(lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")

    #plot_latent_vectors(pmf.users, pmf.items)
    #plot_predicted_ratings(pmf.users, pmf.items)
    #plt.show()

    #pmf.print_latent_vectors()
    #pmf.save_latent_vectors("models/")

if __name__ == '__main__':
    main()
