#!/usr/bin/env python3
'''
Functions to make approximately low-rank matrices with elements from a specific
set and specific properties.
'''

import itertools
import numpy as np
import operator
import random
from collections import Counter


DEF_VALS = (1,2,3,4,5)


def make_orig(m, n, values=DEF_VALS, probs=None):
    if probs is None:
        cdf = np.linspace(0, 1, len(values) + 1)[1:]
    else:
        cdf = np.cumsum(probs)
        cdf /= cdf[-1]
    v = [values[np.searchsorted(cdf, random.random(), side='right')]
            for i in range(m*n)]
    return np.array(v).reshape(m, n)


def low_rank_approx(orig, k):
    u, s, vh = np.linalg.svd(orig)
    v = vh.T
    full_s = np.zeros(orig.shape)
    full_s[range(len(s)), range(len(s))] = s

    return u[:,:k], np.dot(full_s[:k,:k], v[:,:k].T).T


def reconstruct(u, v, vals=DEF_VALS):
    lifted_get = np.vectorize(lambda i: vals[i], otypes=[np.float])
    approx = np.dot(u, v.T)
    return lifted_get(np.argmin([abs(approx-v) for v in vals], axis=0))


def get_counts(ary, vals=DEF_VALS):
    c = Counter(ary.flat)
    return [c[v] for v in vals]


def sample_with_counts(m, n, rank, vals=DEF_VALS, probs=None,
                       min_fracs=.1, max_fracs=.3):
    min_counts = np.array(min_fracs, copy=False) * m*n
    max_counts = np.array(max_fracs, copy=False) * m*n

    if (np.ones(len(vals)) * max_fracs).sum() < 1:
        raise ValueError("not possible to satisfy (maxes too low)")

    while True:
        u, v = low_rank_approx(make_orig(m, n, vals, probs), rank)
        counts = get_counts(reconstruct(u, v, vals))
        if np.all(counts >= min_counts) and np.all(counts <= max_counts):
            return u, v


def sample_with_test(m, n, rank, test, vals=DEF_VALS, probs=None):
    gen = lambda: low_rank_approx(make_orig(m, n, vals, probs), rank)
    uvs = map(operator.methodcaller('__call__'), itertools.repeat(gen))
    return next((u, v) for u, v in uvs if test(u, v))


def has_exact_pos(known, known_pos, unknown_pos, cutoff=4, vals=DEF_VALS):
    unknown = np.logical_not(known)

    if known_pos > known.sum():
        raise ValueError("want more known pos than known points")
    if unknown_pos > unknown.sum():
        raise ValueError("want more unknown pos than unknown points")

    num = 0

    def test(u, v):
        nonlocal num
        num += 1
        if num % 1000 == 0:
            print("test #%d" % num)

        ary = reconstruct(u, v, vals)
        return (ary[known] >= cutoff).sum() == known_pos and \
               (ary[unknown] >= cutoff).sum() == unknown_pos
    return test


def known_diag(m, n):
    known = np.zeros((m, n), dtype=bool)
    indices = np.arange(max(m,n))
    known[indices % m, indices % n] = 1
    return known


def gen_known_diag_counts(m, n, rank, known_pos, unknown_pos,
                          vals=DEF_VALS, prob=None, cutoff=4):
    test = has_exact_pos(known_diag(m, n), known_pos, unknown_pos, cutoff, vals)
    u, v = sample_with_test(m, n, rank, test, vals, prob)
    return reconstruct(u, v, vals)


def main():
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', '-m', type=int, required=True)
    parser.add_argument('--cols', '-n', type=int, required=True)
    parser.add_argument('--rank', '-r', type=int, required=True)
    parser.add_argument('--known-pos', '-k', type=int, required=True)
    parser.add_argument('--unknown-pos', '-K', type=int, required=True)
    parser.add_argument('--cutoff', '-c', type=int, default=4)
    parser.add_argument('--prob', '-p', type=float, nargs='+', default=None)
    parser.add_argument('outfile')

    args = parser.parse_args()

    dirname = os.path.dirname(args.outfile)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    vals = DEF_VALS

    real = gen_known_diag_counts(m=args.rows, n=args.cols, rank=args.rank,
                                 known_pos=args.known_pos,
                                 unknown_pos=args.unknown_pos,
                                 vals=vals, prob=args.prob,
                                 cutoff=args.cutoff)

    known = known_diag(args.rows, args.cols)
    ratings = np.zeros((known.sum(), 3))
    for idx, (i, j) in enumerate(np.transpose(known.nonzero())):
        ratings[idx] = [i, j, real[i,j]]

    data = {
        '_real': real,
        '_ratings': ratings,
        '_rating_vals': vals,
    }

    with open(args.outfile, 'wb') as outfile:
        pickle.dump(data, outfile)


if __name__ == '__main__':
    main()
