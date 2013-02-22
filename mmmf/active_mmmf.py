#!/usr/bin/env python3

from collections import namedtuple
import functools
import os
import random
import shutil
import string
import sys
import subprocess
from tempfile import mkdtemp

import numpy as np
import scipy.io

KeyFunc = namedtuple('KeyFunc', "nice_name code")

KEY_FUNCS = {
    'random': KeyFunc("Random", "@select_random"),
    'min-margin': KeyFunc("Min Margin", "@select_min_margin"),
    'min-margin-pos': KeyFunc("Min Margin Positive", "@select_min_margin_pos"),
    'max-margin': KeyFunc("Max Margin", "@select_max_margin"),
    'max-margin-pos': KeyFunc("Max Margin Positive", "@select_max_margin_pos"),
}


_M_TEMPLATE = '''
dbstop if error
load {infile}

Y = double(Y); % NOTE: true values can't have 0s!
known = known == 1;

selectors = {{ {selectors} }};
C = double(C);

results = evaluate_active(...
    Y, selectors, steps, known, queryable, C, test_on {partial});

save {outfile} results
'''

def compare(keys, data_matrix, known, queryable=None, test_on=None, steps=-1,
            C=1, cutoff=None,
            mat_cmd='matlab',
            return_tempdir=False, delete_tempdir_if_success=True,
            tempdir_base=None, partial_results=True):
    # TODO: switch to fMMMF to not throw out partial solutions

    # # can't contain any zeros
    # if 0 in data_matrix:
    #     data_matrix += .01
    #     assert 0 not in data_matrix

    if not set(data_matrix.flat).issubset([-1, 0, 1]):
        if cutoff is None:
            raise ValueError("we only handle binary matrices here, bud")
        new_data_matrix = np.zeros_like(data_matrix)
        orig_known = np.isfinite(data_matrix) * (data_matrix != 0)
        new_data_matrix[orig_known] = (data_matrix[orig_known] > cutoff) * 2 - 1
        data_matrix = new_data_matrix

    # make temporary dir
    tempdir = mkdtemp(dir=tempdir_base)
    path = functools.partial(os.path.join, tempdir)

    infile_path = path('data_in.mat')
    outfile_path = path('data_out.mat')
    mfile_name = 'run_mfile_' + \
            ''.join(random.choice(string.ascii_letters) for x in range(8))
    mfile_path = path(mfile_name + '.m')

    matdata = {
        'Y': data_matrix,
        'known': known,
        'queryable': queryable,
        'test_on': [] if test_on is None else test_on,
        'steps': steps,
        'C': C,
    }

    if partial_results:
        partial = ", '{}'".format(path('partial_results.mat'))
    else:
        partial = ""

    try:
        scipy.io.savemat(infile_path, matdata, oned_as='column')

        mfile_content = _M_TEMPLATE.format(
            selectors=', '.join(KEY_FUNCS[k].code for k in keys),
            infile=infile_path,
            outfile=outfile_path,
            partial=partial,
        )
        with open(mfile_path, 'w') as mfile:
            mfile.write(mfile_content)

        print(mfile_path)

        # run matlab
        proc = subprocess.Popen([mat_cmd, "-nojvm",
            '-r', "addpath('{}'); {}; exit".format(tempdir, mfile_name)])
        proc.wait()

        # read results
        with open(outfile_path, 'rb') as outfile:
            mat_results = scipy.io.loadmat(outfile)['results']

        if delete_tempdir_if_success:
            shutil.rmtree(tempdir)
    except:
        try:
            proc.kill()
        except:
            pass

        if not return_tempdir:
            shutil.rmtree(tempdir)

        raise

    results = results_from_mat(mat_results, keys)

    if return_tempdir:
        return results, tempdir
    else:
        return results

def _handle_array(array):
    if hasattr(array, 'todense'):
        array = array.todense()
    if array.size:
        array = array.astype(float)
        array[array == 0] = np.nan
    else:
        array = None
    return array

def results_from_mat(mat_results, keys):
    results = {}
    for k, v in zip(keys, mat_results.flat):
        results[k] = [
            [num[0, 0],
             rmse[0, 0],
             (ij[0, 0] - 1, ij[0, 1] - 1) if ij.size else None,
             _handle_array(evals),
             _handle_array(pred),
            ]
            for num, rmse, ij, evals, pred in v
        ]
    return results


def main():
    import argparse
    import pickle

    key_names = KEY_FUNCS.keys()

    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('keys', nargs='*',
            help="Choices: {}.".format(', '.join(sorted(key_names))))

    parser.add_argument('--cutoff', '-c', type=float, default=None)
    parser.add_argument('-C', type=float, default=1)
    parser.add_argument('--steps', '-s', type=int, default=-1)
    parser.add_argument('--data-file', '-D', required=True)
    parser.add_argument('--matlab', '-m', default='matlab')

    g = parser.add_mutually_exclusive_group()
    g.set_defaults(delete_tempdir='success')
    g.add_argument('--delete-tempdir-always', action='store_const',
        dest='delete_tempdir', const='always')
    g.add_argument('--delete-tempdir-on-success', action='store_const',
        dest='delete_tempdir', const='success')
    g.add_argument('--keep-tempdir', action='store_false', dest='delete_tempdir')

    parser.add_argument('--tempdir-base', default=None, metavar='DIR')

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
    real = orig['_real']

    # get sparse matrix of known elements
    known = np.zeros(real.shape, dtype=bool)
    ratings = orig['_ratings']
    known[ratings[:, 0].astype(int), ratings[:, 1].astype(int)] = 1

    test_on = orig.get('_test_on', None)

    queryable = real != 0

    # get new results
    return_tempdir = args.delete_tempdir != 'always'
    delete_tempdir_if_success = bool(args.delete_tempdir)
    results = compare(keys=list(args.keys),
                      data_matrix=real, cutoff=args.cutoff,
                      known=known, queryable=queryable, test_on=test_on,
                      steps=args.steps, C=args.C,
                      mat_cmd=args.matlab,
                      tempdir_base=args.tempdir_base,
                      return_tempdir=return_tempdir,
                      delete_tempdir_if_success=delete_tempdir_if_success,
                      partial_results=False)
    if return_tempdir:
        results, tempdir = results
        if not delete_tempdir_if_success:
            print("Temporary files in {}".format(tempdir))

    # back up original file if we're overwriting it
    if os.path.exists(args.results_file):
        path, name = os.path.split(args.data_file)
        shutil.copy2(args.data_file, os.path.join(path, '.{}.bak'.format(name)))

    # add our results to the original dictionary and save
    orig['_mmmf_args'] = args
    for k, v in results.items():
        orig['mmmf_' + k] = v

    with open(args.results_file, 'wb') as f:
        pickle.dump(orig, f)


if __name__ == '__main__':
    main()
