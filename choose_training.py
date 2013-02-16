#!/usr/bin/env python
from __future__ import division

import argparse
import ast
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
    if num_to_pick is not None:
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

    if num_to_pick is None:
        return known

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


def pick(args, real):
    knowable = np.isfinite(real) & (real != 0)

    if args.pick_no_extras:
        num_to_pick = None
    elif args.n_pick:
        num_to_pick = args.n_pick
    elif args.pick_dataset_frac:
        num_to_pick = int(np.round(real.size * args.pick_dataset_frac))
    else:
        num_to_pick = int(np.round(knowable.sum() * args.pick_known_frac))

    if args.drugbank:
        return pick_ratings_drugbank(real, num_to_pick)
    else:
        return pick_ratings(knowable, num_to_pick)


def figure_out_test(args, real, known):
    knowable = np.isfinite(real) & (real != 0)
    testable = knowable & (~known)

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

        if args.test_class_ratios or args.test_equal_classes:
            labels = list(set(real[knowable].flat))
            n_labels = len(labels)

            if args.test_equal_classes:
                ratios = np.ones(n_labels) / n_labels
            else:
                ratios = np.array([args.test_class_ratios[k] for k in labels])
                total = ratios.sum()
                assert .97 <= total <= 1.03, "total ratio was {}".format(total)
                ratios /= total

            n_per_label = np.round(ratios * num_test).astype(int)
            diff = num_test - n_per_label.sum()
            i = random.sample(sixm.xrange(n_labels), abs(diff))
            n_per_label[i] += np.sign(diff)
            assert n_per_label.sum() == num_test

            test_on = np.zeros(testable.shape, bool)
            for label, num in sixm.zip(labels, n_per_label):
                sample_from_ary((real == label) & testable, test_on, num)

        elif args.test_one_per_row_col:
            test_on = pick_ratings(testable, num_test)

        else:
            test_on = np.zeros(testable.shape, bool)
            sample_from_ary(testable, test_on, num_test)
    return test_on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('outfile')

    parser.add_argument('--drugbank', action='store_true')

    new = parser.add_argument_group('New item options')
    new.add_argument('--know-all-old', action='store_true', default=False)
    g = new.add_mutually_exclusive_group()
    g.add_argument('--n-new-item', type=int, metavar='N')
    g.add_argument('--new-item-frac', type=float, metavar='FRAC')

    initial = parser.add_argument_group('Initially known set options',
        "Chooses this many elements to be initially known, in such a way "
        "that each row and column is picked at least once. Default: "
        "chooses 5% of known ratings.")
    g = initial.add_mutually_exclusive_group()
    g.add_argument('--pick-no-extras', action='store_true')
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
    g.add_argument('--test-class-ratios', type=ast.literal_eval, default=None)

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

    dct = {'_real': real}

    knowable = np.isfinite(real) & (real != 0)

    ### figure out rating_vals
    if real.dtype.kind in 'iu' or \
            np.all(real[knowable] == np.round(real[knowable])):
        # avoid repeat nans in set: http://stackoverflow.com/a/14846245/344821
        rating_set = set(real[~np.isnan(real)])
        rating_set.discard(0)
        dct['_rating_vals'] = tuple(sorted(rating_set))

    n_new = None
    if args.n_new_item:
        n_new = args.n_new_item
    elif args.new_item_frac:
        n_new = int(np.round(real.shape[1] * args.new_item_frac))

    if not n_new:
        known = pick(args, real)
        test_on = figure_out_test(args, real, known)
    else:
        is_new = np.zeros(real.shape[1], dtype=bool)
        is_new[random.sample(sixm.xrange(real.shape[1]), n_new)] = True
        dct['_is_new_item'] = is_new

        if args.know_all_old:
            known_old = knowable[:, ~is_new]
        else:
            known_old = pick(args, real[:, ~is_new])
        known_new = pick(args, real[:, is_new])

        known = np.zeros(real.shape, dtype=bool)
        known[:, ~is_new] = known_old
        known[:, is_new] = known_new

        test_on = np.zeros(real.shape, dtype=bool)
        test_on[:, is_new] = figure_out_test(args, real[:, is_new], known_new)

    dct['_ratings'] = make_ratings(real, known)
    if test_on is not None:
        dct['_test_on'] = test_on

    # save away
    np.savez_compressed(args.outfile, **dct)

if __name__ == '__main__':
    main()
