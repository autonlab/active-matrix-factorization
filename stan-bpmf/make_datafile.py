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
            description='Makes a .Rdata file for the stan bpmf code.')

    parser.add_argument('infile')
    parser.add_argument('outfile')

    parser.add_argument('--rank', '-R', type=int, required=True)
    parser.add_argument('--n-users', default=None, type=int)
    parser.add_argument('--n-items', default=None, type=int)

    # TODO: option to standardize ratings

    args = parser.parse_args()

    infile = np.load(args.infile)
    data = make_vars(infile['_ratings'], rank=args.rank,
                     n_users=args.n_users, n_items=args.n_items)
    with open(args.outfile, 'w') as f:
        dump_to_rdata(output=f, **data)

if __name__ == '__main__':
    main()
