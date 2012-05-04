import numpy as np

def tripexpect(mean, cov, a, b, c):
    '''E[X_a X_b X_c] for N(mean, cov)'''
    return mean[a] * mean[b] * mean[c] + \
            mean[a]*cov[b,c] + mean[b]*cov[a,c] + mean[c]*cov[a,b]

def quadexpect(mean, cov, a, b, c, d):
    '''E[X_a X_b X_c X_d] for N(mean, cov) and distinct a/b/c/d'''
    # TODO: try method of Kan (2008), see if it's faster?
    abcd = [a, b, c, d]
    ma, mb, mc, md = mean[abcd]

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

def exp_squared(mean, cov, a, b):
    '''E[X_a^2 X_b^2] for N(mean, cov)'''
    return 4 * mean[a] * mean[b] * cov[a,b] + 2*cov[a,b]**2 + \
            (mean[a]**2 + cov[a,a]) * (mean[b]**2 + cov[b,b])

def exp_a2bc(mean, cov, a, b, c):
    '''E[X_a^2 X_b X_c for N(mean, cov)'''
    ma, mb, mc = mean[[a,b,c]]
    return (
        (ma**2 + cov[a,a]) * (mb * mc + cov[b,c])
        + 2 * ma * mc * cov[a,b]
        + 2 * ma * mb * cov[a,c]
        + 2 * cov[a,b] * cov[a,c]
    )

def exp_dotprod_sq(u, v, mean, cov, i, j):
    '''E[ (U_i^T V_j)^2 ]'''
    # TODO: vectorize as much as possible
    exp = 0
    for k in range(u.shape[0]):
        uki = u[k,i]
        vkj = v[k,j]

        exp += exp_squared(mean, cov, uki, vkj)

        for l in range(k+1, u.shape[0]):
            exp += 2 * quadexpect(mean, cov, uki, vkj, u[l,i], v[l,j])
    return exp

def normal_gradient(apmf):
    '''
    Find the gradient of the KL divergence w.r.t. to the passed ActivePMF
    model's approximation params.
    '''
    mean = apmf.mean
    cov = apmf.cov
    if mean is None or cov is None:
        raise ValueError("run initialize_approx first")

    u = apmf.u
    v = apmf.v

    us = u.reshape(-1)
    vs = v.reshape(-1)

    sig = apmf.sigma_sq

    # note that we're actually only differentiating by one triangular half
    # of the covariance matrix, but we make grad_cov the full square anyway
    grad_mean = np.zeros_like(mean)
    grad_cov = np.zeros_like(cov)

    def inc_cov_quadexp_grad(a,b, c,d):
        # contribution to grad of sigma_{a,b} by {c,d}, doubled
        inc = np.sum(mean[c] * mean[d] + cov[c, d]) / sig
        grad_cov[a, b] += inc
        grad_cov[b, a] += inc


    for i, j, rating in apmf.ratings:
        # gradient of sum_k sum_{l>k} E[ U_ki V_kj U_li V_lj ] / sigma^2
        # (doubled because of symmetry, cancels with 2 in denom)
        for k in range(apmf.latent_d-1):
            uki = u[k, i]
            vkj = v[k, j]
            # vectorize out our sums over l
            uli = u[k+1:, i]
            vlj = v[k+1:, j]

            grad_mean[uki] += np.sum(tripexpect(mean,cov, vkj,uli,vlj)) / sig
            grad_mean[vkj] += np.sum(tripexpect(mean,cov, uki,uli,vlj)) / sig
            grad_mean[uli] += np.sum(tripexpect(mean,cov, uki,vkj,vlj)) / sig
            grad_mean[vlj] += np.sum(tripexpect(mean,cov, uki,vkj,uli)) / sig

            inc_cov_quadexp_grad(uki,vkj, uli,vlj)
            inc_cov_quadexp_grad(uki,uli, vkj,vlj)
            inc_cov_quadexp_grad(uki,vlj, vkj,uli)
            inc_cov_quadexp_grad(vkj,uli, uki,vlj)
            inc_cov_quadexp_grad(vkj,vlj, uki,uli)
            inc_cov_quadexp_grad(uli,vlj, uki,vkj)

        # everything else can just be vectorized over k
        uki = u[:,i]
        vkj = v[:,j]

        muki = mean[uki]
        mvkj = mean[vkj]

        # gradient of \sum_k E[ U_ki^2 V_kj^2 ] / (2 sigma^2)
        grad_mean[uki] += (2 * mvkj * cov[uki,vkj]
                           + muki * (mvkj**2 + cov[vkj,vkj])) / sig
        grad_mean[vkj] += (2 * muki * cov[uki,vkj]
                           + mvkj * (muki**2 + cov[uki,uki])) / sig

        grad_cov[uki,uki] += (mvkj**2 + cov[vkj,vkj]) / (2*sig)
        grad_cov[vkj,vkj] += (muki**2 + cov[uki,uki]) / (2*sig)

        inc = 2 * (muki * mvkj + cov[uki,vkj]) / sig
        grad_cov[uki,vkj] += inc
        grad_cov[vkj,uki] += inc

        # gradient of - \sum_k Rij E[U_ki V_kj] / sigma^2
        grad_mean[uki] -= mvkj * (rating / sig)
        grad_mean[vkj] -= muki * (rating / sig)

        grad_cov[uki,vkj] -= rating / sig
        grad_cov[vkj,uki] -= rating / sig


    # gradient of \sum_i \sum_k E[U_ki^2] / (2 sigma_u^2), same for V
    grad_mean[us] += mean[us] / apmf.sigma_u_sq
    grad_mean[vs] += mean[vs] / apmf.sigma_v_sq

    grad_cov[us,us] += 1 / (2 * apmf.sigma_u_sq)
    grad_cov[vs,vs] += 1 / (2 * apmf.sigma_v_sq)

    # gradient of ln(|cov|)/2 w.r.t. the triangular half
    #
    # Derivation sketch: the partial of ln(det(sigma)) by sigma[i,j]
    # is the cofactor divided by det(sigma), which is just the [j,i]th
    # element of the inverse (Cramer's rule). But since our matrix is
    # really constrained to be symmetric, we need to use the chain rule
    # to account for the other side of the matrix, which is just
    # the [i,j]th element of the inverse. (For the diagonal, this doesn't
    # come into play, so don't add that on.)
    inv = np.linalg.inv(cov)
    grad_cov += (inv + inv.T * (1 - np.eye(cov.shape[0]))) / 2

    return grad_mean, grad_cov
