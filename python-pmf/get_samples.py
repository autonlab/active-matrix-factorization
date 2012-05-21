#!/usr/bin/env python3
import argparse
import datetime
import gzip
import itertools
import pickle

import numpy as np

from bayes_pmf import BayesianPMF

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('data_file')
arg('out', default='test')

arg('--latent-d', '-D', type=int, default=10)
arg('--batch-size', type=int, default=10000)
arg('--validation-size', type=int, default=10000)
arg('--stop-thresh', type=float, default=1e-3)

args = parser.parse_args()


print("Loading data")
opener = gzip.open if args.data_file.endswith('.gz') else open
with opener(args.data_file, 'rb') as f:
    data = np.load(f)

if isinstance(data, np.ndarray):
    known = data > 0
    ratings = np.zeros((known.sum(), 3))
    for idx, (i, j) in enumerate(np.transpose(known.nonzero())):
        ratings[idx, :] = i, j, data[i, j]
else:
    ratings = data['_ratings']

bpmf = BayesianPMF(ratings, args.latent_d)

print("Doing initial MAP fit")
for train, valid in bpmf.fit_minibatches_until_validation(
        args.batch_size, args.validation_size, do_yield=True, stop_thresh=args.stop_thresh):
    print("\t{} {:.5} {:.5}".format(datetime.datetime.now().time(), train, valid))

print("Saving model")
with open(args.out + '_model.pkl', 'wb') as f:
    pickle.dump(bpmf, f)

print("Getting MCMC samples")
num_samps = 2000
u_samps = np.empty((num_samps, bpmf.num_users, bpmf.latent_d)); u_samps.fill(np.nan)
v_samps = np.empty((num_samps, bpmf.num_items, bpmf.latent_d)); v_samps.fill(np.nan)
for idx, (u, v) in enumerate(itertools.islice(bpmf.samples(), num_samps)):
    if idx % 10 == 0:
        print(datetime.datetime.now().time(), idx)
    u_samps[idx,:,:] = u
    v_samps[idx,:,:] = v

print("Saving u samples")
np.save(args.out + '_u_samps.npy', u_samps)

print("Saving v samples")
np.save(args.out + '_v_samps.npy', v_samps)
