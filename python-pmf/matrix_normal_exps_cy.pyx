import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def tripexpect(np.ndarray[DTYPE_t, ndim=2] mean not None,
               np.ndarray[DTYPE_t, ndim=2] cov_rows not None,
               np.ndarray[DTYPE_t, ndim=2] cov_cols not None,
               object a_i, object a_j,
               object b_i, object b_j,
               object c_i, object c_j):
    '''E[a b c] for MN(mean, cov_rows, cov_cols)'''
    cdef DTYPE_t ma = mean[a_i, a_j]
    cdef DTYPE_t mb = mean[b_i, b_j]
    cdef DTYPE_t mc = mean[c_i, c_j]

    return (ma * mb * mc
           + ma * cov_rows[b_i, c_i] * cov_cols[b_j, c_j]
           + mb * cov_rows[a_i, c_i] * cov_cols[a_j, c_j]
           + mc * cov_rows[a_i, b_i] * cov_cols[a_j, b_j])


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double quadexpect(np.ndarray[DTYPE_t, ndim=2] mean,
                        np.ndarray[DTYPE_t, ndim=2] cov_rows,
                        np.ndarray[DTYPE_t, ndim=2] cov_cols,
                        int a_i, int a_j,
                        int b_i, int b_j,
                        int c_i, int c_j,
                        int d_i, int d_j):
    '''
    E[a b c d] for MN(mean, cov_rows, cov_cols) and distinct a,b,c,d.

    Assumes that a,b,c,d are within bounds and not negative; will do
    nasty things if this isn't true.
    '''
    # TODO: try method of Kan (2008), see if it's faster?

    cdef DTYPE_t ma = mean[a_i, a_j]
    cdef DTYPE_t mb = mean[b_i, b_j]
    cdef DTYPE_t mc = mean[c_i, c_j]
    cdef DTYPE_t md = mean[d_i, d_j]

    cdef DTYPE_t cov_ab = cov_rows[a_i, b_i] * cov_cols[a_j, b_j]
    cdef DTYPE_t cov_ac = cov_rows[a_i, c_i] * cov_cols[a_j, c_j]
    cdef DTYPE_t cov_ad = cov_rows[a_i, d_i] * cov_cols[a_j, d_j]
    cdef DTYPE_t cov_bc = cov_rows[b_i, c_i] * cov_cols[b_j, c_j]
    cdef DTYPE_t cov_bd = cov_rows[b_i, d_i] * cov_cols[b_j, d_j]
    cdef DTYPE_t cov_cd = cov_rows[c_i, d_i] * cov_cols[c_j, d_j]

    return (
        # product of the means
        ma * mb * mc * md

        # pairs of means times cov of the other two
        + ma * mb * cov_cd
        + ma * mc * cov_bd
        + ma * md * cov_bc
        + mb * mc * cov_ad
        + mb * md * cov_ac
        + mc * md * cov_ab

        # pairs of covariances (Isserlis)
        + cov_ab * cov_cd + cov_ac * cov_bd + cov_ad * cov_bc
    )

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double exp_squared(np.ndarray[DTYPE_t, ndim=2] mean,
                         np.ndarray[DTYPE_t, ndim=2] cov_rows,
                         np.ndarray[DTYPE_t, ndim=2] cov_cols,
                         int a_i, int a_j,
                         int b_i, int b_j):
    '''E[a^2 b^2] for MN(mean, cov_rows, cov_cols)

    Assumes that a,b are within bounds and not negative; will do
    nasty things if this isn't true.
    '''
    cdef DTYPE_t ma = mean[a_i, a_j]
    cdef DTYPE_t mb = mean[b_i, b_j]
    cdef DTYPE_t cov_ab = cov_rows[a_i, b_i] * cov_cols[a_j, b_j]
    cdef DTYPE_t var_a = cov_rows[a_i, a_i] * cov_cols[a_j, a_j]
    cdef DTYPE_t var_b = cov_rows[b_i, b_i] * cov_cols[b_j, b_j]

    return (4 * ma * mb * cov_ab
          + 2 * cov_ab ** 2
          + (ma ** 2 + var_a) * (mb ** 2 + var_b))

@cython.boundscheck(False)
@cython.wraparound(False)
def exp_a2bc(np.ndarray[DTYPE_t, ndim=2] mean not None,
             np.ndarray[DTYPE_t, ndim=2] cov_rows not None,
             np.ndarray[DTYPE_t, ndim=2] cov_cols not None,
             int a_i, int a_j,
             int b_i, int b_j,
             int c_i, int c_j):
    '''E[a^2 b c] for MN(mean, cov_rows, cov_cols)

    Assumes that a,b,c are within bounds and not negative; will do
    nasty things if this isn't true.
    '''

    cdef DTYPE_t ma = mean[a_i, a_j]
    cdef DTYPE_t mb = mean[b_i, b_j]
    cdef DTYPE_t mc = mean[c_i, c_j]

    cdef DTYPE_t var_a = cov_rows[a_i, a_i] * cov_cols[a_j, a_j]
    cdef DTYPE_t var_b = cov_rows[b_i, b_i] * cov_cols[b_j, b_j]

    cdef DTYPE_t cov_ab = cov_rows[a_i, b_i] * cov_cols[a_j, b_j]
    cdef DTYPE_t cov_ac = cov_rows[a_i, c_i] * cov_cols[a_j, c_j]
    cdef DTYPE_t cov_bc = cov_rows[b_i, c_i] * cov_cols[b_j, c_j]

    return ((ma ** 2 + var_a) * (mb * mc + cov_bc)
          + 2 * ma * mc * cov_ab
          + 2 * ma * mb * cov_ac
          + 2 * cov_ab * cov_ac)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double exp_dotprod_sq(int num_users,
                            np.ndarray[DTYPE_t, ndim=2] mean,
                            np.ndarray[DTYPE_t, ndim=2] cov_useritems,
                            np.ndarray[DTYPE_t, ndim=2] cov_latents,
                            int i, int j):
    '''E[ (U_i^T V_j)^2 ]
    = E[ (\sum_k U_ik V_jk)^2 ]
    = E[ \sum_k \sum_l U_ik V_jk U_il V_jl ]
    = \sum_k (E[U_ik^2 V_jk^2] + 2 \sum_{l > k} E[U_ik V_jk U_il V_jl])

    Assumes i, j are within bounds and not negative, and that the shapes are
    appopriate; will do nasty things if this isn't true.
    '''
    # TODO: vectorize as much as possible

    cdef double exp = 0
    cdef int latent_dim = mean.shape[1]
    cdef int j_ = num_users + j
    cdef int k, l

    for k in range(latent_dim):
        exp += exp_squared(mean, cov_useritems, cov_latents, i, k,
                                                             j_, k)
        for l in range(k+1, latent_dim):
            exp += 2 * quadexpect(mean, cov_useritems, cov_latents, i, k,
                                                                    j_, k,
                                                                    i, l,
                                                                    j_, l)
    return exp


@cython.boundscheck(False)
@cython.wraparound(False)
def mn_kl_divergence(int num_users,
                     np.ndarray[DTYPE_t, ndim=2] ratings not None,
                     np.ndarray[DTYPE_t, ndim=2] mean not None,
                     np.ndarray[DTYPE_t, ndim=2] cov_useritems not None,
                     np.ndarray[DTYPE_t, ndim=2] cov_latents not None,
                     double sigma_sq,
                     double sigma_u_sq,
                     double sigma_v_sq):
    '''
    KL( MN(mean, cov_useritems, cov_latents)  ||  PMF model ) for given ratings,
    up to an additive constant
    '''
    cdef double kl = 0
    cdef int i, j, rating_idx
    cdef double rating

    cdef int num_useritems = mean.shape[0]
    cdef int num_items = num_useritems - num_useritems
    cdef int latent_d = mean.shape[1]

    # entropy term
    det_sign_useritems, log_det_useritems = np.linalg.slogdet(cov_useritems)
    det_sign_latents, log_det_latents = np.linalg.slogdet(cov_latents)
    kl -= (log_det_useritems * latent_d + log_det_latents * num_useritems) / 2.

    # regularization terms, exploiting tr(A x B) = (tr A) (tr B)
    cdef double tr_cov_latents = cov_latents.trace()

    cdef double tr_cov_users = 0
    cdef double tr_cov_items = 0
    for i in range(num_users):
        tr_cov_users += cov_useritems[i, i]
    for j in range(num_users, num_users + num_items):
        tr_cov_items += cov_useritems[i, i]

    kl += (((mean[:num_users, :] ** 2).sum() + tr_cov_users * tr_cov_latents)
           / (2 * sigma_u_sq))
    kl += (((mean[num_users:, :] ** 2).sum() + tr_cov_items * tr_cov_latents)
           / (2 * sigma_u_sq))

    # the main terms
    cdef double bit = 0
    for rating_idx in range(ratings.shape[0]):
        i = <int> ratings[rating_idx, 0]
        j = <int> ratings[rating_idx, 1]
        rating = ratings[rating_idx, 2]

        bit += exp_dotprod_sq(num_users, mean, cov_useritems, cov_latents, i, j)
        bit -= 2 * rating * (  # - 2 R_ij \sum_k E[U_ki V_kj]
                (mean[i, :] * mean[num_users + j, :]).sum()
                + cov_useritems[i, num_users + j] * tr_cov_latents)
        bit += rating * rating
    kl += bit / (2 * sigma_sq)

    return kl


def matrixnormal_gradient(mn_apmf not None):  # TODO
    '''
    Find the gradient of the KL divergence w.r.t. to the passed MNActivePMF
    model's approximation params.

    Raises TypeError if mn_apmf, .mean, .cov_rows, or .cov_cols is None

    Will do nasty things if the shapes of the various arrays are wrong.
    '''
    if (mn_apmf.mean is None
            or mn_apmf.cov_useritems is None or mn_apmf.cov_latents is None):
        raise TypeError("mean, cov are None; run initialize_approx first")

    cdef np.ndarray mean = mn_apmf.mean
    cdef np.ndarray cov_useritems = mn_apmf.cov_useritems
    cdef np.ndarray cov_latents = mn_apmf.cov_latents

    # note that we're actually only differentiating by one triangular half
    # of the covariance matrix, but we make grad_cov the full square anyway
    cdef np.ndarray g_mean = np.zeros_like(mean)
    cdef np.ndarray g_cov_useritems = np.zeros_like(cov_useritems)
    cdef np.ndarray g_cov_latents = np.zeros_like(cov_latents)

    _mnormal_grad(mean, cov_useritems, cov_latents,
            mn_apmf.ratings, mn_apmf.num_users, mn_apmf.latent_d,
            mn_apmf.sigma_sq, mn_apmf.sigma_u_sq, mn_apmf.sigma_v_sq,
            g_mean, g_cov_useritems, g_cov_latents)

    return g_mean, g_cov_useritems, g_cov_latents


# TODO: vectorize 'n stuff
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _quadexp_grad(int num_users,
                        np.ndarray[DTYPE_t, ndim=2] mean,
                        np.ndarray[DTYPE_t, ndim=2] cov_useritems,
                        np.ndarray[DTYPE_t, ndim=2] cov_latents,
                        np.ndarray[DTYPE_t, ndim=2] g_mean,
                        np.ndarray[DTYPE_t, ndim=2] g_cov_useritems,
                        np.ndarray[DTYPE_t, ndim=2] g_cov_latents,
                        int i, int j, int k, int l,
                        double mult):
    """
    Updates the passed gradients for mult * the gradient of
    E[U_ik V_jk U_il V_jl]
    """
    cdef int j_ = num_users + j
    cdef DTYPE_t inc

    cdef DTYPE_t M_ik = mean[i, k]
    cdef DTYPE_t M_il = mean[i, l]
    cdef DTYPE_t M_jk = mean[j_, k]
    cdef DTYPE_t M_jl = mean[j_, l]

    cdef DTYPE_t cov_ij = cov_useritems[i, j_]
    cdef DTYPE_t var_i = cov_useritems[i, i]
    cdef DTYPE_t var_j = cov_useritems[j_, j_]

    cdef DTYPE_t cov_kl = cov_latents[k, l]
    cdef DTYPE_t var_k = cov_latents[k, k]
    cdef DTYPE_t var_l = cov_latents[l, l]

    g_mean[i, k] += mult * (
          M_jk * M_il * M_jl
        + M_jl * cov_ij * cov_kl
        + M_il * var_j * cov_kl
        + M_jk * cov_ij * var_l
    )
    g_mean[i, l] += mult * (
          M_ik * M_jk * M_jl
        + M_jl * cov_ij * var_k
        + M_jk * cov_ij * cov_kl
        + M_ik * var_j * cov_kl
    )
    g_mean[j_, k] += mult * (
          M_ik * M_il * M_jl
        + M_jl * var_i * cov_kl
        + M_il * cov_ij * cov_kl
        + M_ik * cov_ij * var_l
    )
    g_mean[j_, l] += mult * (
          M_ik * M_jk * M_il
        + M_il * cov_ij * var_k
        + M_jk * var_i * cov_kl
        + M_ik * cov_ij * cov_kl
    )

    g_cov_useritems[i, i] += mult * (
          M_jk * M_jl * cov_kl
        + var_j * cov_kl**2
    )
    g_cov_useritems[j_, j_] += mult * (
        + M_ik * M_il * cov_kl
        + var_i * cov_kl**2
    )
    inc = mult * (
          M_il * M_jl * var_k
        + M_jk * M_il * cov_kl
        + M_ik * M_jl * cov_kl
        + M_ik * M_jk * var_l
        + 2 * cov_ij * var_k * var_l
        + 2 * cov_ij * cov_kl**2
    )
    g_cov_useritems[i, j_] += inc
    g_cov_useritems[j_, i] += inc

    g_cov_latents[k, k] += mult * (
          M_il * M_jl * cov_ij
        + cov_ij**2 * var_l
    )
    g_cov_latents[l, l] += mult * (
          M_ik * M_jk * cov_ij
        + cov_ij**2 * var_k
    )
    inc = mult * (
        + M_jk * M_jl * var_i
        + M_jk * M_il * cov_ij
        + M_ik * M_jl * cov_ij
        + M_ik * M_il * var_j
        + 2 * var_i * var_j * cov_kl
        + 2 * cov_ij**2 * cov_kl
    )
    g_cov_latents[k, l] += inc
    g_cov_latents[l, k] += inc

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _squareexp_grad(int num_users,
                          np.ndarray[DTYPE_t, ndim=2] mean,
                          np.ndarray[DTYPE_t, ndim=2] cov_useritems,
                          np.ndarray[DTYPE_t, ndim=2] cov_latents,
                          np.ndarray[DTYPE_t, ndim=2] g_mean,
                          np.ndarray[DTYPE_t, ndim=2] g_cov_useritems,
                          np.ndarray[DTYPE_t, ndim=2] g_cov_latents,
                          int i, int j, int k,
                          double mult):
    """
    Updates the passed gradients for mult * the gradient of
    E[U_ik^2 V_jk^2]
     = 4 M_ik M_jk Sigma_ij Omega_kk
       + 2 Sigma_ij^2 Omega_kk^2
       + (M_ik^2 + Sigma_ii Omega_kk) (M_jk^2 + Sigma_jj Omega_kk)
    """
    cdef int j_ = j + num_users
    cdef double inc

    cdef DTYPE_t M_ik = mean[i, k]
    cdef DTYPE_t M_jk = mean[j_, k]

    cdef DTYPE_t cov_ij = cov_useritems[i, j_]
    cdef DTYPE_t var_i = cov_useritems[i, i]
    cdef DTYPE_t var_j = cov_useritems[j_, j_]

    cdef DTYPE_t var_k = cov_latents[k, k]

    cdef double e_ik_sq = M_ik * M_ik + var_i * var_k
    cdef double e_jk_sq = M_jk * M_jk + var_j * var_k

    g_mean[i, k] += mult * (4 * M_jk * cov_ij * var_k + 2 * M_ik * e_jk_sq)
    g_mean[j_, k] += mult * (4 * M_ik * cov_ij * var_k + e_ik_sq * 2 * M_jk)

    g_cov_useritems[i, i] += mult * (var_k * e_jk_sq)
    g_cov_useritems[j_, j_] += mult * (e_ik_sq * var_k)

    inc = mult * (4 * (M_ik * M_jk + cov_ij * var_k) * var_k)
    g_cov_useritems[i, j_] += inc
    g_cov_useritems[j_, i] += inc

    g_cov_latents[k, k] += mult * (
          4 * M_ik * M_jk * cov_ij
        + 4 * cov_ij * cov_ij * var_k
        + var_i * e_jk_sq
        + e_ik_sq * var_j)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision
cpdef int _mnormal_grad(np.ndarray[DTYPE_t, ndim=2] mean,
                        np.ndarray[DTYPE_t, ndim=2] cov_useritems,
                        np.ndarray[DTYPE_t, ndim=2] cov_latents,
                        np.ndarray[DTYPE_t, ndim=2] ratings,
                        int num_users, int latent_d,
                        double sig_sq, double sig_u_sq, double sig_v_sq,
                        np.ndarray[DTYPE_t, ndim=2] g_mean,
                        np.ndarray[DTYPE_t, ndim=2] g_cov_useritems,
                        np.ndarray[DTYPE_t, ndim=2] g_cov_latents):
    cdef int i, j, j_, k, l, idx
    cdef np.ndarray mu_i, mv_j, inc
    cdef double rating

    cdef double one_over_sig_sq = 1. / sig_sq
    cdef double mult

    cdef int num_useritems = cov_useritems.shape[0]
    cdef int num_items = num_useritems - num_users

    # the term that uses the ratings...
    for idx in range(ratings.shape[0]):
        i = <int> ratings[idx, 0]
        j = <int> ratings[idx, 1]
        j_ = j + num_users
        rating = ratings[idx, 2]

        # TODO: vectorize

        for k in range(latent_d):
            # gradient of sum_{l>k} E[ U_ik V_jk U_il V_jl ] / sigma^2
            # (doubled because of symmetry, cancels with 2 in denom)
            for l in range(k + 1, latent_d):
                _quadexp_grad(num_users, mean, cov_useritems, cov_latents,
                              g_mean, g_cov_useritems, g_cov_latents,
                              i, j, k, l,
                              one_over_sig_sq)

            # gradient of E[ U_ik^2 V_jk^2 ] / (2 sigma^2)
            _squareexp_grad(num_users, mean, cov_useritems, cov_latents,
                            g_mean, g_cov_useritems, g_cov_latents,
                            i, j, k,
                            one_over_sig_sq / 2)

            # gradient of -R_ij E[U_ik V_jk] / sigma^2
            #   = -R_ij (M_ik M_jk + Sigma_ij Omega_kk) / sigma^2
            mult = -rating * one_over_sig_sq
            g_mean[i, k] += mult * mean[j_, k]
            g_mean[j_, k] += mult * mean[i, k]
            g_cov_useritems[i, j_] += mult * cov_latents[k, k]
            g_cov_useritems[j_, i] += mult * cov_latents[k, k]
            g_cov_latents[k, k] += mult * cov_useritems[i, j_]

    # gradient of \sum_i \sum_k E[U_ik^2] / (2 sigma_u^2), same for V
    #           = \sum_i \sum_k (M_ik^2 + Sigma_ii Omega_kk) / (2 sigma^2)
    g_mean[:num_users, :] += mean[:num_users, :] / sig_u_sq
    g_mean[num_users:, :] += mean[num_users:, :] / sig_v_sq

    tr_latents = cov_latents.trace()
    user_idx = np.arange(num_users)
    item_idx = np.arange(num_items) + num_users
    g_cov_useritems[user_idx, user_idx] += tr_latents / (2 * sig_u_sq)
    g_cov_useritems[item_idx, item_idx] += tr_latents / (2 * sig_v_sq)

    latent_idx = np.arange(latent_d)
    g_cov_latents[latent_idx, latent_idx] += \
            cov_useritems[user_idx, user_idx].sum() / (2 * sig_u_sq)
    g_cov_latents[latent_idx, latent_idx] += \
            cov_useritems[item_idx, item_idx].sum() / (2 * sig_v_sq)

    # gradient of entropy = D/2 ln(det(Sigma)) + (N+M)/2 ln(det(Omega))
    #
    # The gradient of ln(|C|) w.r.t a triangular half is
    # C^-1 + (1 - I) .* C^-T
    #
    # Derivation sketch: the partial of ln(det(sigma)) by sigma[i,j]
    # is the cofactor divided by det(sigma), which is just the [j,i]th
    # element of the inverse (Cramer's rule). But since our matrix is
    # really constrained to be symmetric, we need to use the chain rule
    # to account for the other side of the matrix, which is just
    # the [i,j]th element of the inverse. (For the diagonal, this doesn't
    # come into play, so don't add that on.)

    inv_cov_useritems = np.linalg.inv(cov_useritems)
    g_cov_useritems -= latent_d / 2. * (
        inv_cov_useritems
        + inv_cov_useritems.T * (1 - np.eye(num_useritems)))

    inv_cov_latents = np.linalg.inv(cov_latents)
    g_cov_latents -= num_useritems / 2. * (
        inv_cov_latents
        + inv_cov_latents.T * (1 - np.eye(latent_d)))
