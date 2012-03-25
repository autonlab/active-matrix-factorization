import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# TODO types
def tripexpect(np.ndarray[DTYPE_t, ndim=1] mean not None,
               np.ndarray[DTYPE_t, ndim=2] cov not None,
               a, b, c):
    '''E[X_a X_b X_c] for N(mean, cov)'''
    return mean[a] * mean[b] * mean[c] + \
            mean[a]*cov[b,c] + mean[b]*cov[a,c] + mean[c]*cov[a,b]

@cython.boundscheck(False)
def quadexpect(np.ndarray[DTYPE_t, ndim=1] mean not None,
               np.ndarray[DTYPE_t, ndim=2] cov not None,
               unsigned int a, unsigned int b, unsigned int c, unsigned int d):
    '''
    E[X_a X_b X_c X_d] for N(mean, cov) and distinct a/b/c/d.

    Assumes that a/b/c/d are within bounds and not negative; will do
    nasty things if this isn't true.
    '''
    # TODO: try method of Kan (2008), see if it's faster?

    cdef DTYPE_t ma = mean[a]
    cdef DTYPE_t mb = mean[b]
    cdef DTYPE_t mc = mean[c]
    cdef DTYPE_t md = mean[d]

    return (
        # product of the means
        ma * mb * mc * md

        # pairs of means times cov of the other two
        + ma * mb * cov[c,d]
        + ma * mc * cov[b,d]
        + ma * md * cov[b,c]
        + mb * mc * cov[a,d]
        + mb * md * cov[a,c]
        + mc * md * cov[a,b]

        # pairs of covariances (Isserlis)
        + cov[a,b] * cov[c,d]
        + cov[a,c] * cov[b,d]
        + cov[a,d] * cov[b,c]
    )

def exp_squared(np.ndarray[DTYPE_t, ndim=1] mean not None,
                np.ndarray[DTYPE_t, ndim=2] cov not None,
                int a, int b):
    '''E[X_a^2 X_b^2] for N(mean, cov)'''
    return 4 * mean[a] * mean[b] * cov[a,b] + 2*cov[a,b]**2 + \
            (mean[a]**2 + cov[a,a]) * (mean[b]**2 + cov[b,b])

def exp_a2bc(np.ndarray[DTYPE_t, ndim=1] mean not None,
             np.ndarray[DTYPE_t, ndim=2] cov not None,
             int a, int b, int c):
    '''E[X_a^2 X_b X_c for N(mean, cov)'''

    cdef DTYPE_t ma = mean[a]
    cdef DTYPE_t mb = mean[b]
    cdef DTYPE_t mc = mean[c]

    return (
        (ma**2 + cov[a,a]) * (mb * mc + cov[b,c])
        + 2 * ma * mc * cov[a,b]
        + 2 * ma * mb * cov[a,c]
        + 2 * cov[a,b] * cov[a,c]
    )

def exp_dotprod_sq(np.ndarray[np.int_t, ndim=2] u not None,
                   np.ndarray[np.int_t, ndim=2] v not None,
                   np.ndarray[DTYPE_t, ndim=1] mean not None,
                   np.ndarray[DTYPE_t, ndim=2] cov not None,
                   int i, int j):
    '''E[ (U_i^T V_j)^2 ]'''
    # TODO: vectorize as much as possible

    cdef float exp = 0
    cdef int latent_dim = u.shape[0]

    for k in range(latent_dim):
        uki = u[k,i]
        vkj = v[k,j]

        exp += exp_squared(mean, cov, uki, vkj)

        for l in range(k+1, latent_dim):
            exp += 2 * quadexpect(mean, cov, uki, vkj, u[l,i], v[l,j])
    return exp
