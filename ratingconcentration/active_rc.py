#!/usr/bin/env python3

from collections import namedtuple
import functools
import os
import subprocess
import tempfile

import numpy as np
import scipy.io

# for pickle
import sys
sys.path.append('../python-pmf')
from active_pmf import ActivePMF

KeyFunc = namedtuple('KeyFunc', "nice_name code")

KEY_FUNCS = {
    'ge-2': KeyFunc("Prob >= 2", "select_ge_cutoff(2)"),
    'ge-4': KeyFunc("Prob >= 4", "select_ge_cutoff(4)"),
    'entropy': KeyFunc("Entropy Lookahead", "@select_1step_lowest_entropy"),
}


_M_TEMPLATE = '''
load {infile}
whos

X = double(X); % NOTE: true values can't have 0s!
known = known == 1;

selectors = {{ {selectors} }};

results = evaluate_active(X, known, selectors, steps, delta);

save {outfile} results
'''

def compare(keys, data_matrix, known, steps, delta, mat_cmd='matlab'):
    # TODO: choose delta through CV
    # TODO: control parallelism
    # TODO: get sparse matrices to work

    mattemp = functools.partial(tempfile.NamedTemporaryFile,
            suffix='.mat', delete=False)

    # can't contain any zeros
    if 0 in data_matrix:
        data_matrix += 1
        assert 0 not in data_matrix

    matdata = {
        'X': data_matrix,
        'known': known,
        'steps': steps,
        'delta': delta,
    }

    # make temporary files
    with mattemp(mode='wb') as matfile:
        matfilename = matfile.name
        scipy.io.savemat(matfile, matdata, oned_as='row')

    with mattemp(mode='rb') as outfile:
        outfilename = outfile.name

    mfile_content = _M_TEMPLATE.format(
        infile=matfilename,
        outfile=outfilename,
        selectors=', '.join(KEY_FUNCS[k].code for k in keys),
    )

    try:
        # run matlab
        proc = subprocess.Popen([mat_cmd, "-nojvm"], stdin=subprocess.PIPE)
        proc.communicate(mfile_content.encode())

        # read results
        with open(outfilename, 'rb') as outfile:
            mat_results = scipy.io.loadmat(outfile)['results']
    finally:
        # delete temp files
        os.remove(outfilename)
        os.remove(matfilename)

    results = {}
    for k, v in zip(keys, mat_results):
        results[k] = res = []
        for num, rmse, ij, evals in v[0,]:
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
                (ij[0,0], ij[0,1]) if ij.size else None,
                evals,
            ])

    return results


def main():
    import argparse
    import pickle
    import shutil


    key_names = KEY_FUNCS.keys()

    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('keys', nargs='*',
            help="Choices: {}.".format(', '.join(sorted(key_names))))

    parser.add_argument('--delta', '-d', type=float, default=1.5)
    parser.add_argument('--steps', '-s', type=int, default=-1)
    parser.add_argument('--data-file', '-D', required=True)
    parser.add_argument('--matlab', '-m', default='matlab')

    parser.add_argument('--results-file', default=None, metavar='FILE',
            help="Save results in FILE; by default, add to --data-file.")

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
    with open(args.data_file, 'rb') as f:
        orig = pickle.load(f)
    
    # get sparse matrtix of known elements
    known = np.zeros(orig['_real'].shape, dtype=bool)
    ratings = orig['_ratings']
    known[ratings[:,0].astype(int), ratings[:,1].astype(int)] = 1

    # get new results
    results = compare(args.keys, orig['_real'], known, args.steps, args.delta,
                      args.matlab)

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
