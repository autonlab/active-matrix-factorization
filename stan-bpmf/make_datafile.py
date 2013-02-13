#!/usr/bin/env python

import numpy as np
from dump_rdata import dump_to_rdata

# needed variables:
#   n_users, n_items, rank
#   n_obs, obs_users[], obs_items[], obs_ratings[]
#   rating_std (2)
#   mu_0[] (0)
#   beta_0 (2), nu_0 (rank), w_0 (I)

def make_vars(ratings, rank, n_users=None, n_items=None):
    users = ratings[:, 0].astype(int) + 1
    items = ratings[:, 1].astype(int) + 1

    assert np.all(users - 1 == ratings[:, 0])
    assert np.all(items - 1 == ratings[:, 1])

    if n_users: assert users.max() <= n_users
    if n_items: assert items.max() <= n_items

    data = {
        'rank': rank,
        'n_users': n_users or users.max(),
        'n_items': n_items or items.max(),
        'n_obs': ratings.shape[0],
        'obs_users': users,
        'obs_items': items,
        'obs_ratings': ratings[:, 2],

        'rating_std': 2,
        'mu_0': np.zeros(rank),
        'beta_0': 2,
        'nu_0': rank,
        'w_0': np.eye(rank),
    }
    return data

def main():
    import argparse
    parser = argparse.ArgumentParser(
            description='Makes a data file for the stan bpmf code.')

    parser.add_argument('infile')
    parser.add_argument('outfile')

    parser.add_argument('--rank', '-R', type=int, required=True)
    parser.add_argument('--n-users', default=None, type=int)
    parser.add_argument('--n-items', default=None, type=int)

    parser.add_argument('--subtract-mean', action='store_true', default=False)

    parser.add_argument('--output-format', default='numpy',
        choices=['rdata', 'matlab', 'numpy'])

    # TODO: option to standardize ratings

    args = parser.parse_args()

    infile = np.load(args.infile)
    ratings = infile['_ratings']
    if args.subtract_mean:
        ratings[:, 2] -= np.mean(ratings[:, 2])

    data = make_vars(ratings, rank=args.rank,
                     n_users=args.n_users, n_items=args.n_items)

    if args.output_format == 'rdata':
        if not args.outfile.endswith('.rdata'):
            args.outfile += '.rdata'
        with open(args.outfile, 'w') as f:
            dump_to_rdata(output=f, **data)
    elif args.output_format == 'matlab':
        from scipy.io import savemat
        savemat(args.outfile, data, oned_as='column')
    elif args.output_format == 'numpy':
        np.savez(args.outfile, **data)
    else:
        raise ValueError

if __name__ == '__main__':
    main()
