#!/usr/bin/env python3

import gzip
import pickle
import random

import numpy as np

import sys
sys.path.append('../python-pmf')
import get_criteria

outfile = sys.argv[1]
num_pos = int(sys.argv[2])
num_neg = int(sys.argv[3])
assert num_neg >= 1
infile = sys.argv[4] if len(sys.argv) > 4 else 'top_hundred.npy'

with (gzip.open if infile.endswith('.gz') else open)(infile) as f:
    real = np.load(f).astype(int) * 2 - 1 # turn zeros into negatives

assert set(real.flat) == {-1, 1}

known = np.zeros(real.shape, bool)

# choose the passed number of positives
positives = (real == 1).reshape(-1)
pos_indices = list(positives.nonzero()[0])
pos_picked = random.sample(pos_indices, num_pos)
known.flat[pos_picked] = 1

# choose the passed number of negatives
known[-1,-1] = 1
neg_indices = list(np.logical_not(positives).nonzero()[0])
neg_picked = random.sample(neg_indices, num_neg-1)
known.flat[neg_picked] = 1


# make the ratings matrix, rating_vals
ratings = get_criteria.make_ratings(real, known)
rating_vals = tuple(sorted(set(real.flat) - {0, np.nan}))


dct = {'_real': real, '_ratings': ratings, '_rating_vals': rating_vals}

# save away
with open(outfile, 'wb') as f:
    pickle.dump(dct, f)
