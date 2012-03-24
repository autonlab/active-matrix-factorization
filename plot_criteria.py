#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt

from active_pmf import ActivePMF, plot_criteria, plot_predictions, \
                       make_fake_data, KEY_FUNCS

print("generating data")
real, _ = make_fake_data(.25, 5, 5, 0, 3)

mask = np.array([
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
])

ratings = np.zeros((mask.sum(), 3))
for idx, (i, j) in enumerate(np.transpose(mask.nonzero())):
    ratings[idx] = [i, j, real[i, j]]

print("fitting pmf")
apmf = ActivePMF(ratings, 2)
apmf.fit()
apmf.initialize_approx()
apmf.fit_normal()

if __name__ == '__main__':
    print("plotting predictions")
    plt.figure()
    plot_predictions(apmf, real)

    print("plotting criteria")
    plt.figure()
    plot_criteria(apmf, [KEY_FUNCS[i] for i in
        ('random', 'pred-variance',
            #'total-variance', 'total-variance-approx',
            #'uv-entropy', 'uv-entropy-approx',
         'pred-entropy-bound', 'pred-entropy-bound-approx')],
        procs=None)
    plt.show()
