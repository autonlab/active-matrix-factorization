import gzip
import os
import pickle

import numpy as np

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

# numpy interaction
from rpy2.robjects.numpy2ri import numpy2ri
def numpy2ri_avoiding_zerodim(x):
    if hasattr(x, 'shape') and x.shape == ():
        x = x.dtype.type(x)
    return numpy2ri(x)
ro.conversion.py2ri = numpy2ri_avoiding_zerodim

# make inline happy
if list(ro.r('Sys.getenv("R_ARCH")'))[0] == '':
    arch = list(ro.r('.Platform$r_arch'))[0]
    ro.r('Sys.setenv(R_ARCH="/{}")'.format(arch))

rstan = importr('rstan')


# TODO: stan_model, stanfit class wrappers


def get_model(filename, cache_filename=None, check_times=True, use_cache=True):
    '''
    Returns a stan_model for the model code in filename.
    If use_cache (by default), tries to load the compiled file from
    cache_filename (default filename + '.model.pkl.gz') if available,
    otherwise compiles it and saves into the gzipped, pickled cache file.
    '''
    if cache_filename is None and use_cache:
        cache_filename = filename + '.model.pkl.gz'

    if use_cache and os.path.exists(cache_filename) and (not check_times or
            os.path.getmtime(cache_filename) >= os.path.getmtime(filename)):
        try:
            with gzip.open(cache_filename, 'rb') as f:
                return pickle.load(f)
        except (EOFError, IOError):
            pass

    model = rstan.stan_model(file=filename)
    if use_cache:
        with gzip.open(cache_filename, 'wb') as f:
            pickle.dump(model, f)
    return model


def sample(model, data, par_names=None, return_fit=False, **fit_params):
    '''
    Samples from the model (returned from get_model) with the given data.
    Any kwargs are passed on to the sampling() function from rstan.

    If par_names is a list of parameter names, returns a tuple of sample arrays
    (one array per parameter, first dimension the number of samples.)

    If par_names is None, returns that tuple for all params, and also the tuple
    of parameter names, in a list.

    If return_fit, returns the stanfit object as the last element of the
    returned list.
    '''
    fit = rstan.sampling(model, data=ro.r.list(**data), verbose=True, **fit_params)

    args = {'pars': par_names} if par_names else {}
    samples = ro.r.extract(fit, permuted=True, **args)

    ret = [tuple(np.asarray(x) for x in samples)]
    if not par_names:
        ret.append(tuple(samples.names))
    if return_fit:
        ret.append(fit)

    return ret[0] if len(ret) == 1 else ret
