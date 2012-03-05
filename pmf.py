#!/usr/bin/env python
'''
An implementation of Probabilistic Matrix Factorization:
    Salakhutdinov, R., & Mnih, A. (2007). Probabilistic Matrix Factorization.
    Advances in Neural Information Processing Systems.

Loosely based on code by Danny Tarlow:
http://blog.smellthedata.com/2009/06/netflix-prize-tribute-recommendation.html
'''

from __future__ import division

import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import random


class ProbabilisticMatrixFactorization(object):
    def __init__(self, rating_tuples, latent_d=1):
        self.latent_d = latent_d
        self.learning_rate = .0001

        self.sigma_sq = 1
        self.sigma_u_sq = 10
        self.sigma_v_sq = 10

        self.ratings = np.array(rating_tuples, dtype=float)
        if self.ratings.shape[1] == 3:
            self.ratings = np.hstack((self.ratings,
                                      np.ones((len(rating_tuples), 1))))
        if self.ratings.shape[1] != 4:
            raise TypeError("invalid rating tuple length")

        self.rated = set((i, j) for i, j, rating, weight in self.ratings)

        self.num_users = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = int(np.max(self.ratings[:, 1]) + 1)

        self.users = np.random.random((self.num_users, self.latent_d))
        self.items = np.random.random((self.num_items, self.latent_d))


    def add_rating(self, i, j, rating, weight=1):
        self.add_ratings([i, j, rating, weight])

    def add_ratings(self, extra):
        rows, cols = self.ratings.shape

        extra = np.array(extra, copy=False, ndmin=2)
        if len(extra.shape) == 2:
            if extra.shape[1] != cols:
                raise TypeError("bad shape for extra")
        else:
            raise TypeError("bad shape for extra")

        assert np.max(extra[:,0] + 1) <= self.num_users
        assert np.max(extra[:,1] + 1) <= self.num_items

        self.ratings = np.append(self.ratings, extra, 0)
        # TODO: this can be done faster with .resize()...
        #       but the commented-out version is way broken
        # new_rows = extra.shape[0]
        # try:
        #     self.ratings.resize(rows + new_rows, cols)
        # except ValueError:
        #     self.ratings = self.ratings.copy()
        #     self.ratings.resize(rows + new_rows, cols)
        # self.ratings[rows:, :] = extra

        self.rated.update((int(i), int(j)) for i, j in extra[:,:2])


    def log_likelihood(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        sq_error = 0
        for i, j, rating, weight in self.ratings:
            r_hat = np.dot(users[i], items[j])
            sq_error += weight * (rating - r_hat)**2

        return (- sq_error / (2 * self.sigma_sq)
                - np.linalg.norm(users) / (2 * self.sigma_u_sq)
                - np.linalg.norm(items) / (2 * self.sigma_v_sq))


    def try_updates(self, grad_u, grad_v):
        lr = self.learning_rate
        new_users = (1 - lr / (2*self.sigma_u_sq)) * self.users + lr * grad_u
        new_items = (1 - lr / (2*self.sigma_v_sq)) * self.items + lr * grad_v

        return new_users, new_items


    def fit_lls(self):
        old_ll = self.log_likelihood()
        converged = False

        while not converged:
            # calculate the gradient
            grad_u = np.zeros((self.num_users, self.latent_d))
            grad_v = np.zeros((self.num_items, self.latent_d))
            for i, j, rating, wt in self.ratings:
                r_hat = np.dot(self.users[i], self.items[j])
                grad_u[i, :] += self.items[j, :] * ((rating - r_hat) * wt)
                grad_v[j, :] += self.users[i, :] * ((rating - r_hat) * wt)

            # take one step, trying different learning rates if necessary
            while True:
                #print "  setting learning rate =", self.learning_rate
                new_users, new_items = self.try_updates(grad_u, grad_v)
                new_ll = self.log_likelihood(new_users, new_items)

                if new_ll > old_ll:
                    self.users = new_users
                    self.items = new_items
                    self.learning_rate *= 1.25

                    if new_ll - old_ll < .1:
                        converged = True
                    break
                else:
                    self.learning_rate *= .5

                    if self.learning_rate < 1e-10:
                        converged = True
                        break

            yield new_ll
            old_ll = new_ll


    def fit(self):
        for ll in self.fit_lls():
            pass


    def print_latent_vectors(self):
        print "Users:"
        for i in range(self.num_users):
            print "%d: %s" % (i, self.users[i, :])

        print "\nItems:"
        for j in range(self.num_items):
            print "%d: %s" % (j, self.items[j, :])


    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent_d)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent_d)


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
        for j in random.sample(xrange(num_items), num_ratings):
            rating = np.dot(u[i], v[j]) + np.random.normal(scale=noise)
            ratings.append((i, j, rating))

    return (np.array(ratings), u, v)


def plot_ratings(ratings):
    pylab.plot(ratings[:, 1], ratings[:, 2], 'bx')
    pylab.show()


def plot_latent_vectors(U, V):
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
    u_rows, u_cols = U.shape
    v_rows, v_cols = V.shape

    r_hats = -5 * np.ones((u_rows + u_cols + 1, v_rows + v_cols + 1))

    for i in xrange(u_rows):
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


if __name__ == "__main__":
    DATASET = 'fake'

    if DATASET == 'fake':
        (ratings, true_o, true_d) = fake_ratings()

    #plot_ratings(ratings)

    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=5)

    lls = []
    for ll in pmf.fit_lls():
        lls.append(ll)
        print "L=", ll

    plt.figure()
    plt.plot(lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")

    plot_latent_vectors(pmf.users, pmf.items)
    plot_predicted_ratings(pmf.users, pmf.items)
    plt.show()

    pmf.print_latent_vectors()
    #pmf.save_latent_vectors("models/")
