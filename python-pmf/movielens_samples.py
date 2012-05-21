import datetime
import gzip
import pickle

import numpy as np

from bayes_pmf import BayesianPMF

movielens = np.load(gzip.open('../movielens/ratings_matrix.npy.gz'))

known = movielens > 0
ratings = np.zeros((known.sum(), 3))
for idx, (i, j) in enumerate(np.transpose(known.nonzero())):
    ratings[idx, :] = i, j, movielens[i, j]

bpmf = BayesianPMF(ratings, 10)

print("Doing initial MAP fit")
for train, valid in bpmf.fit_minibatches_until_validation(10000, 10000, do_yield=True, stop_thresh=1e-3):
    print("\t{} {.5} {.5}".format(datetime.now().time(), train, valid))

print("Saving model")
with open('movilens_model.pkl', 'wb') as f:
    pickle.dump(bpmf, f)

print("Getting MCMC samples")
num_samps = 2000
u_samps = np.empty((num_samps, bpmf.num_users, bpmf.latent_d)); u_samps.fill(np.nan)
v_samps = np.empty((num_samps, bpmf.num_items, bpmf.latent_d)); v_samps.fill(np.nan)
for idx, (u, v) in enumerate(islice(bpmf.samples(), num_samps)):
    if idx % 10 == 0:
        print(datetime.datetime.now().time(), idx)
    u_samps[idx,:,:] = u
    v_samps[idx,:,:] = v

print("Saving u samples")
np.save('movielens_u_samps.npy', u_samps)

print("Saving v samples")
np.save('movielens_v_samps.npy', v_samps)
