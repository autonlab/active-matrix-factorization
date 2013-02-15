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


def pick_ratings_drugbank(real, num_to_pick):
    knowable = np.isfinite(real)
    assert knowable.sum() > num_to_pick

    pos = knowable & (real > 0)
    neg = knowable & (real <= 0)

    n_drugs, n_targets = knowable.shape
    known = np.zeros(knowable.shape, bool)

    # choose a positive for each drug
    for i in range(n_drugs):
        j = random.choice(list(pos[i, :].nonzero()[0]))
        known[i, j] = 1
        knowable[i, j] = 0

    # choose a negative for any empty targets
    for j in np.logical_not(known.sum(axis=0)).nonzero()[0]:
        i = random.choice(list(neg[:, j].nonzero()[0]))
        known[i, j] = 1
        knowable[i, j] = 0

    assert known.sum() < num_to_pick

    # choose the rest as random negatives
    knowable_negatives = list(neg.ravel().nonzero()[0])
    num_to_pick -= known.sum()
    picked = random.sample(knowable_negatives, num_to_pick)
    known.flat[picked] = 1

    return known


def sample_from_ary(available, target, num):
    avail_pts = list(sixm.zip(*available.nonzero()))
    picked = random.sample(avail_pts, num)
    target[tuple(np.transpose(picked))] = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('outfile')

    parser.add_argument('--drugbank', action='store_true')

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
    g.add_argument('--test-at-random', action='store_true', default=True)
    g.add_argument('--test-equal-classes', action='store_true', default=False)

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

    if args.drugbank:
        real = real.astype(np.int8)
        real[real == 0] = -1

    knowable = np.isfinite(real) & (real != 0)

    if args.n_pick:
        num_to_pick = args.n_pick
    elif args.pick_dataset_frac:
        num_to_pick = int(np.round(real.size * args.pick_dataset_frac))
    else:
        num_to_pick = int(np.round(knowable.sum() * args.pick_known_frac))

    if args.drugbank:
        known = pick_ratings_drugbank(real, num_to_pick)
    else:
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

        if args.test_equal_classes:
            assert np.all(real[knowable] == np.round(real[knowable]))
            labels = list(set(real[knowable].flat))
            n_labels = len(labels)

            n_per_label = np.ones(n_labels, dtype=int) * (num_test // n_labels)
            i = random.sample(sixm.xrange(n_labels), num_test % n_labels)
            n_per_label[i] += 1

            test_on = np.zeros(testable.shape, bool)
            for label, num in sixm.zip(labels, n_per_label):
                sample_from_ary((real == label) & testable, test_on, num)

        elif args.test_one_per_row_col:
            test_on = pick_ratings(testable, num_test)

        else:
            test_on = np.zeros(testable.shape, bool)
            sample_from_ary(testable, test_on, num_test)

    # make the ratings matrix, rating_vals
    ratings = make_ratings(real, known)

    # avoid repeat nans in set: http://stackoverflow.com/a/14846245/344821
    rating_set = set(real[~np.isnan(real)])
    rating_set.discard(0)
    rating_vals = tuple(sorted(rating_set))

    dct = {'_real': real, '_ratings': ratings, '_rating_vals': rating_vals}
    if test_on is not None:
        dct['_test_on'] = test_on

    # save away
    np.savez_compressed(args.outfile, **dct)

if __name__ == '__main__':
    main()
