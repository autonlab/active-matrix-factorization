#!/usr/bin/env python3

import gzip
import pickle
import random

import numpy as np

import sys
sys.path.append('../python-pmf')
import get_criteria

outfile = sys.argv[1]
num_pos = orig_num_pos = int(sys.argv[2])
num_neg = orig_num_neg = int(sys.argv[3])
infile = sys.argv[4] if len(sys.argv) > 4 else 'top_hundred.npy'

with (gzip.open if infile.endswith('.gz') else open)(infile, 'rb') as f:
    real = np.load(f).astype(int) * 2 - 1 # turn zeros into negatives

assert set(real.flat) == {-1, 1}

known = np.zeros(real.shape, bool)

# choose one positive per row
positives = (real == 1)
for row in range(real.shape[0]):
    known[row, random.choice(positives[row].nonzero()[0])] = 1
    num_pos -= 1

# choose the passed number of extra positives
pos_indices = list((positives * np.logical_not(known)).nonzero()[0])
if num_pos < 0:
    raise ValueError("asked for too few positives ({})".format(num_pos))
elif num_pos > 0:
    pos_picked = random.sample(pos_indices, num_pos)
    known.flat[pos_picked] = 1

# make sure each column has at least one known
negatives = np.logical_not(positives) * np.logical_not(known)
zero_cols = known.sum(axis=0) == 0
for col in zero_cols.nonzero()[0]:
    known[random.choice(negatives[:,col].nonzero()[0]), col] = 1
    num_neg -= 1

assert np.all(known.sum(axis=0) > 0)
assert np.all(known.sum(axis=1) > 0)

# choose the passed number of extra negatives
unknown_negs = np.logical_not(positives) * np.logical_not(known)
neg_indices = list(unknown_negs.reshape(-1).nonzero()[0])
neg_picked = random.sample(neg_indices, num_neg)
known.flat[neg_picked] = 1

# make the ratings matrix, rating_vals
ratings = get_criteria.make_ratings(real, known)
rating_vals = tuple(sorted(set(real.flat) - {0, np.nan}))

assert (ratings[:,2] == 1).sum() == orig_num_pos
assert (ratings[:,2] == -1).sum() == orig_num_neg

dct = {'_real': real, '_ratings': ratings, '_rating_vals': rating_vals}

# save away
with open(outfile, 'wb') as f:
    pickle.dump(dct, f)
