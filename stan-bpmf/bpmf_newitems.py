#!/usr/bin/env python
from __future__ import division, print_function

from collections import namedtuple
import os

import numpy as np

import bpmf


class NewItemsBPMF(bpmf.BPMF):
    def __init__(self, new_item_rating_tuples, latent_d,
                 user_factors, fixed_item_factors,
                 model_filename='bpmf_newitems.stan',
                 **kwargs):
        assert user_factors.shape[1] == fixed_item_factors.shape[1] == latent_d

        super(NewItemsBPMF, self).__init__(
            new_item_rating_tuples, latent_d, **kwargs)
        assert user_factors.shape[0] == self.num_users

        self.user_factors = user_factors
        self.fixed_item_factors = fixed_item_factors
        self.num_fixed_items = fixed_item_factors.shape[0]
        self.model_filename = model_filename or 'bpmf_newitems.stan'

    def _data_for_sampling(self):
        data = super(NewItemsBPMF, self)._data_for_sampling()
        data['n_new_items'] = data.pop('n_items')
        data['n_fixed_items'] = self.num_fixed_items
        data['U'] = self.user_factors
        data['V_fixed'] = self.fixed_item_factors
        return data

    def _fill_predictions(self, samps):
        samps['predictions'] = np.einsum('ij,akj->aik',
            self.user_factors, samps['V_new'])


def jigger_ratings(ratings, include):
    new_ratings = ratings[include[ratings[:, 1].astype(int)], :]
    old_to_new_idx = np.cumsum(include, dtype=int) - 1
    new_ratings[:, 1] = old_to_new_idx[new_ratings[:, 1].astype(int)]
    return new_ratings


class MainProgram(bpmf.MainProgram):
    def get_parser(self):
        parser = super(MainProgram, self).get_parser()
        g = parser.add_argument_group('New Items setup')
        g.add_argument('--initial-fit-file', default=None)
        g.add_argument('--initial-fit-samps', default=200, type=int)
        g.add_argument('--initial-fit-warmup', default=200, type=int)
        g.add_argument('--initial-fit-only', action='store_true', default=False)
        return parser

    def do_initial_fit(self, ratings, args):
        model = bpmf.BPMF(ratings, latent_d=args.latent_d,
                          subtract_mean=args.subtract_mean)
        samps = model.samples(num_samps=args.initial_fit_samps,
                              warmup=args.initial_fit_warmup)
        return [np.mean(samps[k], axis=0) for k in ('U', 'V')]
        # TODO: what if it's multimodal? ...

    def load_data(self, args):
        # load data
        with open(args.load_data, 'rb') as f:
            data = np.load(f)

            real = data['_real']
            ratings = data['_ratings']
            is_new_item = data['_is_new_item']
            rating_vals = data['_rating_vals'] if '_rating_vals' in data else None
            test_on = data['_test_on'] if '_test_on' in data else None

        ratings = np.asarray(ratings)

        if args.initial_fit_file and os.path.exists(args.initial_fit_file):
            with open(args.initial_fit_file, 'rb') as f:
                initial_fit = np.load(f)

                user_factors = initial_fit['user_factors']
                fixed_item_factors = initial_fit['fixed_item_factors']
                rank = args.latent_d
                assert user_factors.shape[1] == rank
                assert fixed_item_factors.shape == ((~is_new_item).sum(), rank)
            print("Loaded initial fit from '{}'".format(args.initial_fit_file))
        else:
            print("Doing initial fit...")
            old_r = jigger_ratings(ratings, ~is_new_item)
            user_factors, fixed_item_factors = self.do_initial_fit(old_r, args)
            print("Done with initial fit.")

            if args.initial_fit_file:
                np.savez(args.initial_fit_file,
                         user_factors=user_factors,
                         fixed_item_factors=fixed_item_factors)
                if args.initial_fit_only:
                    import sys
                    sys.exit()

        if args.test_set_from_file and (test_on is not None):
            test_set = test_on
        else:
            try:
                test_set = int(args.test_set)
            except ValueError:
                try:
                    test_set = float(args.test_set)
                except ValueError:
                    test_set = args.test_set

        if args.discrete is None:
            args.discrete = rating_vals is not None

        Data = namedtuple("Data", "real ratings rating_vals test_set "
                                "user_factors fixed_item_factors is_new_item")
        return Data(real[:, is_new_item],
                    jigger_ratings(ratings, is_new_item),
                    rating_vals,
                    test_set[:, is_new_item],
                    user_factors, fixed_item_factors,
                    is_new_item)

    def initialize_bpmf(self, args, data, query_set):
        return NewItemsBPMF(data.ratings, args.latent_d,
                user_factors=data.user_factors,
                fixed_item_factors=data.fixed_item_factors,
                subtract_mean=args.subtract_mean,
                rating_values=data.rating_vals,
                discrete_expectations=args.discrete,
                num_integration_pts=args.num_integration_pts,
                knowable=query_set,
                model_filename=args.model_filename)

if __name__ == '__main__':
    MainProgram().main()
