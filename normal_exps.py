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
