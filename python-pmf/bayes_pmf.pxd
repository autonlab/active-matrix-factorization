cimport cython
cimport numpy as np

from pmf_cy cimport ProbabilisticMatrixFactorization

@cython.locals(n=cython.int, chol=np.ndarray, X=np.ndarray, Y=np.ndarray)
cpdef np.ndarray sample_wishart(np.ndarray sigma, int dof)


@cython.locals(count=cython.int)
cpdef object iter_mean(object iterable)


cdef class BayesianPMF(ProbabilisticMatrixFactorization):
    cdef public float beta
    cdef public tuple _rating_values
    cdef public np.ndarray _rating_bounds
    cdef public bint discrete_expectations
    cdef public int num_integration_pts
    cdef public tuple u_hyperparams, v_hyperparams

    #cpdef BayesianPMF __copy__(self)
    #cpdef BayesianPMF __deepcopy__(self, object memodict)
    cpdef dict __getstate__(self)

    @cython.locals(
        wi = np.ndarray,
        b0 = int,
        df = int,
        mu0 = np.ndarray,
        N = int,
        x_bar = np.ndarray,
        S_bar = np.ndarray,
        mu0_xbar = np.ndarray,
        WI_post = np.ndarray,
        alpha = np.ndarray,
        mu_temp = np.ndarray,
        lam = np.ndarray,
        mu = np.ndarray)
    cpdef tuple sample_hyperparam(self, np.ndarray feats, bint do_users)

    @cython.locals(
        rated_feats = np.ndarray,
        cov = np.ndarray,
        mean = np.ndarray,
        lam = np.ndarray)
    cpdef np.ndarray sample_feature(self, int n, bint is_user,
            np.ndarray mu, np.ndarray alpha, np.ndarray oth_feats,
            np.ndarray rated_indices, np.ndarray ratings)

    # samples() described in the .py file, because I can't get it to work here

    cpdef np.ndarray matrix_results(self, object vals, object which)

    # seems to break everything
    #cpdef np.ndarray predict(self, object samples_iter, object which=*)

    @cython.locals(vals=cython.list)
    cpdef np.ndarray pred_variance(self, object samples_iter, object which=*)

    #cpdef np.ndarray total_variance(self, object samples_iter, object which=*)

    cpdef np.ndarray exp_variance(self, object samples_iter, object which=*,
            object pool=*, object fit_first=*, int num_samps=*)

    @cython.locals(
        n = cython.int,
        m = cython.int,
        all_indices = np.ndarray,
        indices = np.ndarray,
        j_indices = np.ndarray,
        i_indices = np.ndarray,
        vals = np.ndarray,
        discrete = bint,
        alpha = cython.float,
        prev_samps = cython.int,
        denom = cython.float,
        mean = np.ndarray,
        var = np.ndarray,
        res = np.ndarray,
        idx = cython.int,
        exp = cython.float,
    )
    cpdef np.ndarray _distribute(self, object fn, object samples_iter,
            object which, object pool, object fit_first, int num_samps)

    @cython.locals(shape=np.ndarray, counts=np.ndarray, num=cython.int)
    cpdef np.ndarray prob_ge_cutoff(self,
            object samples_iter, float cutoff, object which=*)

    @cython.locals(shape=tuple)
    cpdef np.ndarray random(self, object samples_iter, object which=*)

    @cython.locals(pred=np.ndarray)
    cpdef float bayes_rmse(self, object samples_iter,
                           np.ndarray true_r, object which=*)


#cpdef float _integrate_lookahead(object fn, BayesianPMF bpmf, int i, int j,
#        bint discrete, object params, object fit_first, int num_samps)
