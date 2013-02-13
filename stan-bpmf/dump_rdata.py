#!/usr/bin/env python

import numpy as np
import sys

from six import iteritems
from six.moves import zip as izip
from six.moves import xrange
from itertools import chain, repeat, islice

def intersperse(delimiter, seq):
    # http://stackoverflow.com/a/5655803/344821
    return islice(chain.from_iterable(izip(repeat(delimiter), seq)), 1, None)

def _write_vec(vec, output):
    output.write('c(')
    for thing in intersperse(', ', vec):
        output.write(str(thing))
    output.write(')')

def _write_rep(val, output):
    if np.isscalar(val) and np.isreal(val):
        output.write(str(val))
    elif isinstance(val, xrange) and abs(val[0] - val[1]) == 1:
        output.write('{}:{}'.format(val[0], val[-1]))
    else:
        mat = np.asarray(val)
        if mat.ndim == 1:
            _write_vec(mat, output)
        elif mat.ndim > 1:
            output.write('structure(')
            _write_vec(mat.T.flat, output)
            output.write(', .Dim = ')
            _write_vec(mat.shape, output)
            output.write(')')
        else:
            raise TypeError("Don't know how to handle data {!r}".format(val))

def dump_to_rdata(output=sys.stdout, **things):
    assert hasattr(output, 'write')

    for name, val in iteritems(things):
        output.write(name)
        output.write(' <- ')
        _write_rep(val, output)
        output.write('\n')

def main():
    import argparse
    from scipy.io import loadmat

    parser = argparse.ArgumentParser(description='Converts mat file to Rdata.')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    def _handle(val):
        v = np.asarray(val)
        if v.shape == (1,1):
            return v[0, 0]
        elif v.ndim == 2 and (v.shape[0] == 1 or v.shape[1] == 1):
            return v.reshape(-1)
        return val

    try:
        input_data = loadmat(args.input)
    except IOError:
        input_data = np.load(args.input)

    data = {k: _handle(v)
            for k, v in iteritems(input_data)
            if not k.startswith('__')}

    with open(args.output, 'w') as f:
        dump_to_rdata(output=f, **data)

if __name__ == '__main__':
    main()
