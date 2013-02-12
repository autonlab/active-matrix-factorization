#!/usr/bin/env python

import argparse
import gzip
import random

import numpy as np
import six.moves as sixm


def make_ratings(real, known):
    ratings = np.zeros((known.sum(), 3))
    for idx, (i, j) in enumerate(np.transpose(known.nonzero())):
        ratings[idx] = [i, j, real[i, j]]
    return ratings


def pick_ratings(knowable, num_to_pick):
    assert knowable.sum() > num_to_pick
    knowable = knowable.copy()

    known = np.zeros(knowable.shape, bool)

    # choose a rating from each column
    for j in np.logical_not(known.sum(axis=0)).nonzero()[0]:
        i = random.choice(list(knowable[:, j].nonzero()[0]))
        known[i, j] = 1
        knowable[i, j] = 0

    # choose a rating from each empty row
    for i in np.logical_not(known.sum(axis=1)).nonzero()[0]:
        j = random.choice(list(knowable[i, :].nonzero()[0]))
        known[i, j] = 1
        knowable[i, j] = 0

    assert known.sum() < num_to_pick

    # choose the rest to round it out to num_to_pick chosen things
    knowable_indices = list(knowable.ravel().nonzero()[0])
    num_to_pick -= known.sum()
    picked = random.sample(knowable_indices, num_to_pick)
    known.flat[picked] = 1

    return known


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='50x50_rowscols.npy',
        help="A numpy ratings matrix; default %(default)s.")
    parser.add_argument('outfile')

    initial = parser.add_argument_group('Initially known set options',
        "Chooses this many elements to be initially known, in such a way "
        "that each row and column is picked at least once. Default: "
        "chooses 5% of known ratings.")
    g = initial.add_mutually_exclusive_group()
    g.add_argument('--n-pick', type=int, metavar='N')
    g.add_argument('--pick-dataset-frac', type=float, metavar='FRAC')
    g.add_argument('--pick-known-frac', type=float, metavar='FRAC', default=0.05)

    test = parser.add_argument_group('Test set options',
        "Chooses this many elements for a test set. Default: doesn't pick one. "
        "Does not overlap with the initially-known set.")
    g = test.add_mutually_exclusive_group()
    g.add_argument('--test-one-per-row-col', action='store_true', default=False)
    g.add_argument('--test-at-random', action='store_false',
                   dest='test_one_per_row_col')
    g = test.add_mutually_exclusive_group()
    g.add_argument('--n-test', type=int, metavar='N')
    g.add_argument('--test-dataset-frac', type=float, metavar='FRAC',
        help="Test on FRAC of the overall ratings matrix (regardless of how "
             "many ratings are known).")
    g.add_argument('--test-known-frac', type=float, metavar='FRAC',
        help="Test on FRAC of the knowable ratings.")
    g.add_argument('--test-knowable-frac', type=float, metavar='FRAC',
        help="Test on FRAC of the knowable ratings left after choosing the "
             "initially known values.")

    args = parser.parse_args()

    try:
        with gzip.GzipFile(args.file, 'rb') as f:
            real = np.load(f)
    except IOError:
        real = np.load(args.file)
    knowable = np.isfinite(real) & (real > 0)

    if args.n_pick:
        num_to_pick = args.n_pick
    elif args.pick_dataset_frac:
        num_to_pick = int(np.round(real.size * args.pick_dataset_frac))
    else:
        num_to_pick = int(np.round(knowable.sum() * args.pick_known_frac))
    known = pick_ratings(knowable, num_to_pick)
    testable = knowable & ~known

    num_test = None
    if args.n_test:
        num_test = args.n_test
    elif args.test_dataset_frac:
        num_test = int(np.round(real.size * args.pick_dataset_frac))
    elif args.test_known_frac:
        num_test = int(np.round(knowable.sum() * args.pick_known_frac))
    elif args.test_knowable_frac:
        num_test = int(np.round(testable.sum() * args.pick_knowable_frac))

    test_on = None
    if num_test:
        assert num_test < testable.sum()

        if args.test_one_per_row_col:
            test_on = pick_ratings(testable, num_test)
        else:
            avail_pts = list(sixm.zip(*testable.nonzero()))
            picked_indices = random.sample(avail_pts, num_test)
            test_on = np.zeros(testable.shape, bool)
            test_on[tuple(np.transpose(picked_indices))] = 1

    # make the ratings matrix, rating_vals
    ratings = make_ratings(real, known)
    rating_vals = tuple(sorted(set(real.flat) - set((0, np.nan))))

    dct = {'_real': real, '_ratings': ratings, '_rating_vals': rating_vals}
    if test_on is not None:
        dct['_test_on'] = test_on

    # save away
    np.savez_compressed(args.outfile, **dct)

if __name__ == '__main__':
    main()
