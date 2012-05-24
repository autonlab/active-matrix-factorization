cimport numpy as np

cdef class ProbabilisticMatrixFactorization:
    cdef public int latent_d, num_users, num_items
    cdef public bint subtract_mean
    cdef public double mean_rating
    cdef public double learning_rate, min_learning_rate, stop_thresh
    cdef public double sigma_sq, sigma_u_sq, sigma_v_sq
    cdef public double sig_u_mean, sig_u_var, sig_v_mean, sig_v_var
    cdef public np.ndarray ratings, users, items
    cdef public set rated, unrated

    cpdef ProbabilisticMatrixFactorization __copy__(self)
    cpdef ProbabilisticMatrixFactorization __deepcopy__(self, object memodict)

    cpdef double prediction_for(self, int i, int j,
            np.ndarray users=*, np.ndarray items=*) except? 1492

    cpdef double log_likelihood(self, np.ndarray users=*,
                                      np.ndarray items=*) except? 1492

    cpdef double ll_prior_adjustment(self) except? 1492
    cpdef double full_ll(self, users=*, items=*) except? 1492

    cpdef tuple gradient(self, np.ndarray ratings=*)

    cpdef update_sigma(self)
    cpdef update_sigma_uv(self)

    cpdef np.ndarray predicted_matrix(self, np.ndarray u=*, np.ndarray v=*)

    cpdef double rmse(self, np.ndarray real) except -1
