#!/usr/bin/env python3

from collections import namedtuple
import functools
import os
import random
import shutil
import string
import subprocess
from tempfile import mkdtemp

import numpy as np
import scipy.io

# for pickle
import sys
sys.path.append('../python-pmf')
from active_pmf import ActivePMF

KeyFunc = namedtuple('KeyFunc', "nice_name code")

KEY_FUNCS = {
    'ge-1': KeyFunc("Prob >= 1", "select_ge_cutoff(1)"),
    'ge-4': KeyFunc("Prob >= 4", "select_ge_cutoff(4)"),
    'entropy': KeyFunc("Entropy Lookahead", "@select_1step_lowest_entropy"),
    'random': KeyFunc("Random", "@select_random"),
}


_M_TEMPLATE = '''
dbstop if error
load {infile}

X = double(X); % NOTE: true values can't have 0s!
known = known == 1;
vals = double(vals);

selectors = {{ {selectors} }};

results = evaluate_active(X, known, selectors, steps, delta, vals, pred_mode);

save {outfile} results
'''

def compare(keys, data_matrix, known, steps, delta, pred_mode=False,
            mat_cmd='matlab', return_tempdir=False, vals=None):
    # TODO: choose delta through CV
    # TODO: control parallelism
    # TODO: get sparse matrices to work

    # can't contain any zeros
    if 0 in data_matrix:
        data_matrix += .01
        assert 0 not in data_matrix


    # make temporary dir
    tempdir = mkdtemp()
    path = functools.partial(os.path.join, tempdir)

    infile_path = path('data_in.mat')
    outfile_path = path('data_out.mat')
    mfile_name = 'run_mfile_' + \
            ''.join(random.choice(string.ascii_letters) for x in range(8))
    mfile_path = path(mfile_name + '.m')

    matdata = {
        'X': data_matrix,
        'known': known,
        'steps': steps,
        'delta': delta,
        'vals': vals if vals is not None else sorted(set(data_matrix.flat)),
        'pred_mode': pred_mode,
    }

    try:
        scipy.io.savemat(infile_path, matdata, oned_as='column')

        mfile_content = _M_TEMPLATE.format(
            selectors=', '.join(KEY_FUNCS[k].code for k in keys),
            infile=infile_path,
            outfile=outfile_path,
        )
        with open(mfile_path, 'w') as mfile:
            mfile.write(mfile_content)

        # run matlab
        proc = subprocess.Popen([mat_cmd, "-nojvm",
            '-r', "addpath('{}'); {}; exit".format(tempdir, mfile_name)])
        proc.wait()

        # read results
        with open(outfile_path, 'rb') as outfile:
            mat_results = scipy.io.loadmat(outfile)['results']
    finally:
        if not return_tempdir:
            shutil.rmtree(tempdir)

    results = results_from_mat(mat_results, keys)

    if return_tempdir:
        return results, tempdir
    else:
        return results

def results_from_mat(mat_results, keys):
    results = {}
    for k, v in zip(keys, mat_results.flat):
        results[k] = res = []
        for num, rmse, ij, evals in v:
            if evals.size:
                if hasattr(evals, 'todense'):
                    evals = evals.todense()
                evals = evals.astype(float)
                evals[evals == 0] = np.nan
            else:
                evals = None

            res.append([
                num[0,0],
                rmse[0,0],
                (ij[0,0]-1, ij[0,1]-1) if ij.size else None,
                evals,
            ])
    return results


def main():
    import argparse
    import pickle

    key_names = KEY_FUNCS.keys()

    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('keys', nargs='*',
            help="Choices: {}.".format(', '.join(sorted(key_names))))

    parser.add_argument('--delta', '-d', type=float, default=1.5)
    parser.add_argument('--pred-mode', action='store_true', default=False)
    parser.add_argument('--pred-mean', action='store_false', dest='pred_mode')
    parser.add_argument('--steps', '-s', type=int, default=-1)
    parser.add_argument('--data-file', '-D', required=True)
    parser.add_argument('--matlab', '-m', default='matlab')
    parser.add_argument('--delete-tempdir', action='store_true', default=True)
    parser.add_argument('--no-delete-tempdir',
            action='store_false', dest='delete_tempdir')

    parser.add_argument('--results-file', '-R', default=None, metavar='FILE',
            help="Save results in FILE; by default, add to --data-file.")
    parser.add_argument('--note', action='append',
        help="Doesn't do anything, just there to save any notes you'd like "
             "in the results file.")

    args = parser.parse_args()

    # check that args.keys are valid
    for k in args.keys:
        if k not in key_names:
            sys.stderr.write("Invalid key name %s; options are %s.\n" % (
                k, ', '.join(sorted(key_names))))
            sys.exit(1)

    if not args.keys:
        args.keys = sorted(key_names)

    # save into original file by default
    if args.results_file is None:
        args.results_file = args.data_file

    # load data
    orig = dict(**np.load(args.data_file))

    # get sparse matrtix of known elements
    known = np.zeros(orig['_real'].shape, dtype=bool)
    ratings = orig['_ratings']
    known[ratings[:,0].astype(int), ratings[:,1].astype(int)] = 1

    # get new results
    results = compare(keys=list(args.keys), data_matrix=orig['_real'],
                      known=known, steps=args.steps, delta=args.delta,
                      pred_mode=args.pred_mode,
                      mat_cmd=args.matlab,
                      return_tempdir=not args.delete_tempdir,
                      vals=orig.get('_rating_vals'))
    if not args.delete_tempdir:
        results, tempdir = results
        print("Temporary files in {}".format(tempdir))

    # back up original file if we're overwriting it
    if os.path.exists(args.results_file):
        path, name = os.path.split(args.data_file)
        shutil.copy2(args.data_file, os.path.join(path, '.{}.bak'.format(name)))

    # add our results to the original dictionary and save
    orig['_rc_args'] = args
    for k, v in results.items():
        orig['rc_' + k] = v

    with open(args.results_file, 'wb') as f:
        pickle.dump(orig, f)


if __name__ == '__main__':
    main()
