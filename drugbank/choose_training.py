#!/usr/bin/env python3

import gzip
import pickle
import random

import numpy as np

import sys
sys.path.append('../python-pmf')
import get_criteria

outfile = sys.argv[1]
infile = sys.argv[3] if len(sys.argv) > 3 else 'top_hundred.npy'

with (gzip.open if infile.endswith('.gz') else open)(infile) as f:
    real = np.load(f)

if len(sys.argv) > 2:
    num_to_pick = int(sys.argv[2])
else:
    num_to_pick = int(np.round(real.size * .05))

# assume all non-entries are negatives
assert set(real.flat) == {0, 1}
real[real == 0] = -1

knowable = real != 0
known = np.zeros(real.shape, bool)

# choose at least one rating from every row and column
for j in np.logical_not(known.sum(axis=0)).nonzero()[0]:
    i = random.choice(list(knowable[:,j].nonzero()[0]))
    known[i, j] = 1
    knowable[i, j] = 0

for i in np.logical_not(known.sum(axis=1)).nonzero()[0]:
    j = random.choice(list(knowable[i,:].nonzero()[0]))
    known[i, j] = 1
    knowable[i, j] = 0

# choose the rest to round it out to x% of the full dataset
knowable_indices = list(knowable.reshape(-1).nonzero()[0])
num_to_pick -= known.sum()
picked = random.sample(knowable_indices, num_to_pick)
known.flat[picked] = 1

# make the ratings matrix, rating_vals
ratings = get_criteria.make_ratings(real, known)
rating_vals = tuple(sorted(set(real.flat) - {0, np.nan}))


dct = {'_real': real, '_ratings': ratings, '_rating_vals': rating_vals}

# save away
with open(outfile, 'wb') as f:
    pickle.dump(dct, f)
