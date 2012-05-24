from distutils.core import setup
from distutils.extension import Extension

import numpy as np

setup(
    include_dirs = [np.get_include()],
    ext_modules = [
        Extension("normal_exps_cy", ["normal_exps_cy.c"]),
        Extension("pmf_cy", ["pmf_cy.c"]),
        Extension("bayes_pmf", ["bayes_pmf.c"]),
    ]
)
