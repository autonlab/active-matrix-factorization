import numpy as np
from functools import partial

import normal_exps as pure
import normal_exps_cy as cyth
from active_pmf import project_psd

NUM_SAMPS = 5 * 10**5
NUM_TIMES = 3

def _mean(iterable):
    count = 0
    total = 0
    for x in iterable:
        count += 1
        total += x
    return total / count


def check_expectation(dim, exp_fn, monte_fn=np.prod,
                      num_times=NUM_TIMES, num_samps=NUM_SAMPS):
    for x in range(num_times):
        mn = np.random.normal(0, 10, (dim,))
        cov = project_psd(np.random.normal(0, 5, (dim,dim)), 1e-5)
        samps = np.random.multivariate_normal(mn, cov, num_samps)

        monte = _mean(monte_fn(row) for row in samps)
        exp1 = getattr(pure, exp_fn)(mn, cov, *range(dim))
        exp2 = getattr(cyth, exp_fn)(mn, cov, *range(dim))

        assert exp1 - exp2 < 1e-7
        assert abs(monte - exp1) / exp1 < .02, "%r - %r" % (monte, exp1)

def test_tripexpect():
    check_expectation(3, 'tripexpect')

def test_quadexpect():
    check_expectation(4, 'quadexpect')

def test_exp_squared():
    check_expectation(2, 'exp_squared', lambda row: (row ** 2).prod())

def test_exp_a2bc():
    check_expectation(3, 'exp_a2bc', lambda row: row[0]**2 * row[1] * row[2])

def test_exp_dotprod_sq():
    n = 1
    m = 1
    d = 3
    k = (n + m) * d
    u = np.arange(0, n * d).reshape(n, d).T
    v = np.arange(n * d, (n+m) * d).reshape(m, d).T
    for i in range(NUM_TIMES):
        mn = np.random.normal(0, 10, (k,))
        cov = project_psd(np.random.normal(0, 5, (k,k)), 1e-5)
        samps = np.random.multivariate_normal(mn, cov, NUM_SAMPS)

        monte = _mean(np.dot(x[u[:,0]], x[v[:,0]])**2 for x in samps)
        exp1 = pure.exp_dotprod_sq(u, v, mn, cov, 0, 0)
        exp2 = cyth.exp_dotprod_sq(u, v, mn, cov, 0, 0)

        assert exp1 - exp2 < 1e-7
        assert abs(monte - exp1) / exp1 < .02, "%r - %r" % (monte, exp1)
