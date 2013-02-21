#!/usr/bin/env python3
'''
An implementation of Probabilistic Matrix Factorization:
    Salakhutdinov, R., & Mnih, A. (2007). Probabilistic Matrix Factorization.
    Advances in Neural Information Processing Systems.

Loosely based on code by Danny Tarlow:
http://blog.smellthedata.com/2009/06/netflix-prize-tribute-recommendation.html
'''

import itertools
import random

import numpy as np

def rmse(exp, obs):
    return np.sqrt(np.mean((obs - exp) ** 2))

def rmse_on(exp, obs, on):
    return np.sqrt(np.mean((obs[on] - exp[on]) ** 2))

class ProbabilisticMatrixFactorization(object):
    def __init__(self, rating_tuples, latent_d=1, subtract_mean=False,
                 knowable=None, fit_type=('batch',)):
        self.latent_d = latent_d
        self.subtract_mean = subtract_mean

        self.learning_rate = 1e-4
        self.min_learning_rate = 1e-10
        self.stop_thresh = 1e-2
        self.fit_type = fit_type

        self.sigma_sq = 1
        self.sigma_u_sq = 10
        self.sigma_v_sq = 10

        # negative variance means not to use a prior here
        # would set them to None, but that doesn't work in cython
        self.sig_u_mean = self.sig_v_mean = 0
        self.sig_u_var = self.sig_v_var = -1

        self.ratings = np.array(rating_tuples, dtype=float, copy=False)
        if self.ratings.shape[1] != 3:
            raise TypeError("invalid rating tuple length")
        self.mean_rating = np.mean(self.ratings[:,2])

        self.num_users = n = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = m = int(np.max(self.ratings[:, 1]) + 1)

        self.rated = set((i, j) for i, j, rating in self.ratings)
        if knowable is None:
            knowable = itertools.product(range(n), range(m))
        self.unrated = set(knowable).difference(self.rated)

        self.users = np.random.random((n, self.latent_d))
        self.items = np.random.random((m, self.latent_d))

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__

    def add_rating(self, i, j, rating):
        self.add_ratings([i, j, rating])

    def add_ratings(self, extra):
        rows, cols = self.ratings.shape

        extra = np.array(extra, copy=False, ndmin=2)
        if len(extra.shape) != 2 or extra.shape[1] != cols:
            raise TypeError("bad shape for extra")

        assert np.max(extra[:, 0] + 1) <= self.num_users
        assert np.max(extra[:, 1] + 1) <= self.num_items

        rating_values = getattr(self, 'rating_values', None)
        if rating_values is not None:
            if not set(rating_values).issuperset(extra[:, 2]):
                raise ValueError("got ratings with bad values")

        new_items = set((int(i), int(j)) for i, j in extra[:, :2])

        if not new_items.isdisjoint(self.rated):
            raise ValueError("can't rate already rated items")
        self.rated.update(new_items)
        self.unrated.difference_update(new_items)

        self.ratings = np.append(self.ratings, extra, 0)
        self.mean_rating = np.mean(self.ratings[:, 2])
        # TODO: this can be done without a copy by .resize()...

    def prediction_for(self, i, j, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        if self.subtract_mean:
            return np.dot(users[i], items[j]) + self.mean_rating
        else:
            return np.dot(users[i], items[j])

    def log_likelihood(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
        predfor = self.prediction_for

        sq_error = 0
        for i, j, rating in self.ratings:
            r_hat = predfor(i, j, users, items)
            sq_error += (rating - r_hat) ** 2

        user_norm2 = np.sum(users * users)
        item_norm2 = np.sum(items * items)

        return (- sq_error / (2 * self.sigma_sq)
                - user_norm2 / (2 * self.sigma_u_sq)
                - item_norm2 / (2 * self.sigma_v_sq))

    def ll_prior_adjustment(self):
        return - .5 * (
                np.log(self.sigma_sq) * self.ratings.shape[0]
                + self.num_users * self.latent_d * np.log(self.sigma_u_sq)
                + self.num_items * self.latent_d * np.log(self.sigma_v_sq))

    def full_ll(self, users=None, items=None):
        return self.log_likelihood(users, items) + self.ll_prior_adjustment()

    def gradient(self, ratings=None):
        if ratings is None:
            ratings = self.ratings

        users = self.users
        items = self.items
        sig = self.sigma_sq
        predfor = self.prediction_for

        grad_u = -users / self.sigma_u_sq
        grad_v = -items / self.sigma_v_sq

        for i, j, rating in ratings:
            r_hat = predfor(i, j, users, items)
            grad_u[i, :] += items[j, :] * ((rating - r_hat) / sig)
            grad_v[j, :] += users[i, :] * ((rating - r_hat) / sig)

        return grad_u, grad_v

    def update_sigma(self):
        sq_error = 0
        for i, j, rating in self.ratings:
            r_hat = self.prediction_for(i, j)
            sq_error += (rating - r_hat)**2

        self.sigma_sq = sq_error / self.ratings.shape[0]

    def update_sigma_uv(self):
        d = self.latent_d
        n = self.num_users
        m = self.num_items

        user_norm2 = np.sum(self.users * self.users)
        item_norm2 = np.sum(self.users * self.users)

        if self.sig_u_var > 0:
            self.sigma_u_sq = user_norm2 / (n * d + 2 +
                2*(np.log(self.sigma_u_sq) - self.sig_u_mean) / self.sig_u_var)
        else:
            self.sigma_u_sq = user_norm2 / n / d

        if self.sig_v_var > 0:
            self.sigma_v_sq = item_norm2 / (m * d + 2 +
                2*(np.log(self.sigma_v_sq) - self.sig_v_mean) / self.sig_v_var)
        else:
            self.sigma_v_sq = item_norm2 / m / d

    def fit_lls(self):
        lr = self.learning_rate

        old_ll = self.log_likelihood()

        converged = False
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

    def fit_minibatches(self, batch_size, lr=1, momentum=.8, ratings=None):
        # NOTE: this randomly shuffles ratings / self.ratings
        if ratings is None:
            ratings = self.ratings
        num_ratings = ratings.shape[0]

        # if it's not evenly divisible, we'll have one smaller batch
        batches = list(range(0, num_ratings, batch_size)) + [num_ratings]

        u_inc = np.zeros((self.num_users, self.latent_d))
        v_inc = np.zeros((self.num_items, self.latent_d))

        while True:
            np.random.shuffle(ratings)

            # follow gradient on minibatches
            for batch_start, batch_end in zip(batches[:-1], batches[1:]):
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

    def fit_minibatches_validation(self, batch_size, valid_size, **kwargs):
        total = self.ratings.shape[0]
        valid_idx = set(random.sample(range(total), valid_size))
        train_idx = tuple(i for i in range(total) if i not in valid_idx)
        train = self.ratings[train_idx, :]

        valid_idx = list(valid_idx)
        valid_ijs = tuple(self.ratings[valid_idx, :2].T.astype(int))
        valid_real = self.ratings[valid_idx, 2]

        for train_err in self.fit_minibatches(batch_size, ratings=train,
                                              **kwargs):
            valid_pred = self.predicted_matrix()[valid_ijs]
            valid_err = np.sqrt(np.mean((valid_pred - valid_real) ** 2))
            yield train_err, valid_err

    def fit_minibatches_until_validation(self, *args, stop_thresh=1e-3, **kw):
        last_valid = np.inf
        for train, valid in self.fit_minibatches_validation(*args, **kw):
            if valid > last_valid - stop_thresh:
                break
            last_valid = valid

    def fit_with_sigmas_lls(self, noise_every=10, users_every=5):
        cont = True
        while cont:
            cont = False
            for i, ll in enumerate(self.fit_lls()):
                if i % noise_every == 0:
                    self.update_sigma()
                if i % users_every == 0:
                    self.update_sigma_uv()

                yield ll

                cont = True  # continue if we made any steps

            self.update_sigma()
            self.update_sigma_uv()

    def fit_with_sigmas(self, noise_every=10, users_every=5):
        for ll in self.fit_with_sigmas_lls(noise_every, users_every):
            pass

    def predicted_matrix(self, u=None, v=None):
        if u is None:
            u = self.users
        if v is None:
            v = self.items

        pred = np.dot(u, v.T)
        if self.subtract_mean:
            pred += self.mean_rating
        return pred

    def rmse(self, real, on=None):
        if on is None:
            return rmse(self.predicted_matrix(), real)
        else:
            return rmse_on(self.predicted_matrix(), real, on)

    def print_latent_vectors(self):
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

    plot_latent_vectors(pmf.users, pmf.items)
    plot_predicted_ratings(pmf.users, pmf.items)
    plt.show()

    pmf.print_latent_vectors()
    #pmf.save_latent_vectors("models/")

if __name__ == '__main__':
    main()
