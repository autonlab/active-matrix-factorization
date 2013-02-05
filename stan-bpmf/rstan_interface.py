from functools import partial
import gzip
import os
import pickle
import sys

import numpy as np
import six

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

# numpy interaction
from rpy2.robjects.numpy2ri import numpy2ri
def numpy2ri_avoiding_zerodim(x):
    if hasattr(x, 'shape') and x.shape == ():
        # cast into normal python scalar...sigh
        kinds = {
            'b': bool,
            'u': int,
            'i': int,
            'f': float,
            'c': complex,
        }
        try:
            x = kinds[x.dtype.kind](x)
        except KeyError:
            pass  # just pass it along
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
    cache_filename (default filename + '.model.pkl[2|3].gz') if available,
    otherwise compiles it and saves into the gzipped, pickled cache file.
    '''
    if cache_filename is None and use_cache:
        cache_filename = '{}.model.pkl{}.gz'.format(
            filename, sys.version_info[0])

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
            pickle.dump(model, f, protocol=2)
    return model


class OutputCapturer(object):
    def __init__(self, do_stdout=True, do_stderr=True, merge=False):
        try:
            from StringIO import StringIO
        except ImportError:
            from io import StringIO

        self.do_stdout = do_stdout
        self.do_stderr = do_stderr
        self.merge = merge

        if do_stdout and do_stderr and merge:
            self.stdout = self.stderr = StringIO()
        else:
            if do_stdout:
                self.stdout = StringIO()
            if do_stderr:
                self.stderr = StringIO()

        if self.do_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = self.stdout
        if self.do_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = self.stderr

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.do_stdout:
            sys.stdout = self.old_stdout
        if self.do_stderr:
            sys.stderr = self.old_stderr

        if self.do_stdout and self.do_stderr and self.merge:
            self.stdout.close()
        else:
            if self.do_stdout:
                self.stdout.close()
            if self.do_stderr:
                self.stderr.close()


def sample(model, data, par_names=None, return_fit=False,
           eat_output=False, return_output=True, **fit_params):
    '''
    Samples from the model (returned from get_model) with the given data.
    Any kwargs are passed on to the sampling() function from rstan.

    Returns a dictionary mapping parameter names to sample arrays (whose first
    dimension is the number of samples). The par_names argumet limits which
    parameters are put in the dictionary.

    If return_fit, returns a pair of the dictionary above and the stanfit
    object.

    If eat_output, grabs output from Stan instead of printing it. If
    return_output is also true (default), returns the output string as the last
    element of the return array.
    '''
    if 'warmup' in fit_params and 'iter' in fit_params:
        assert fit_params['iter'] > fit_params['warmup']

    if 'init' in fit_params:
        init = fit_params['init']
        if init is None:
            del fit_params['init']
        elif isinstance(init, six.string_types) or init == 0 or callable(init):
            pass  # do callables work?
        else:
            n_chains = fit_params.get('chains', 4)
            fit_params['init'] = [ro.r.list(**init)] * n_chains

    eat = eat_output
    with OutputCapturer(do_stdout=eat, do_stderr=eat, merge=True) as cap:
        fit = rstan.sampling(model, data=ro.r.list(**data), **fit_params)
        if eat_output:
            output = cap.stdout.getvalue()
    # TODO: handle error conditions better here

    extract = partial(ro.r.extract, fit, permuted=True)
    if par_names:
        samples = extract(pars=par_names)
    else:
        samples = extract()
        par_names = samples.names
    samps = {k: np.asarray(v) for k, v in zip(par_names, samples)}

    ret = [samps]
    if return_fit:
        ret.append(fit)
    if eat_output and return_output:
        ret.append(output)
    return ret[0] if len(ret) == 1 else ret
