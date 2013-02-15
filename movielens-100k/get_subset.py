#!/usr/bin/env python
from __future__ import print_function, division

import gzip
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data-file', default='ratings_matrix.npy.gz')
parser.add_argument('--users-portion', default=0.5, type=float)
parser.add_argument('--movies-rankings-portion', default=0.9, type=float)
parser.add_argument('outfile')
args = parser.parse_args()

n = args.data_file
with (gzip.GzipFile if n.endswith('.gz') else open)(n) as f:
    data = np.load(f)

print("Original data:", data.shape)

# take e.g. half the users
user_rankings = np.sum(data > 0, axis=1)
num_to_take = int(np.ceil(args.users_portion * len(user_rankings)))
cutoff = np.sort(user_rankings)[-num_to_take]
new = data[user_rankings >= cutoff, :]

print("Taking the top {:.0%} of users:".format(args.users_portion), new.shape)

# take the movies that comprise e.g. 90% of the total rankings
movie_rankings = np.sum(new > 0, axis=0)
rev_sorted_rankings = -np.sort(-movie_rankings)
cdf = np.cumsum(rev_sorted_rankings) / movie_rankings.sum()
idx = np.searchsorted(cdf, args.movies_rankings_portion)
new = new[:, movie_rankings >= rev_sorted_rankings[idx]]

print("Taking movies to make {:.0%} of the ratings:"
        .format(args.movies_rankings_portion), new.shape)

# remove any users that now don't have any rankings
new = new[np.any(new, axis=1), :]

print("After removing any zero-rating users:", new.shape)

np.save(args.outfile, new)
