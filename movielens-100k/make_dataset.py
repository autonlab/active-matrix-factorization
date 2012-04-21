#!/usr/bin/env python3

import numpy as np
import random

def make_dataset(ratings, common_movies, n_users=10, n_movies=10):
    users = random.sample(range(ratings.shape[0]), n_users)
    movies = random.sample(list(common_movies), n_movies)
    return ratings[np.ix_(users, movies)], movies

def main(source, target):
    orig = np.load(source)
    new, movies = make_dataset(orig['ten_biggest'], orig['common_movies'])
    np.savez_compressed(target,
        _real=new,
        _rating_vals=[1,2,3,4,5],
        _movie_ids=movies,
    )

if __name__ == '__main__':
    import sys
    main(sys.argv[2] if len(sys.argv) > 2 else 'biggest.npz', sys.argv[1])
