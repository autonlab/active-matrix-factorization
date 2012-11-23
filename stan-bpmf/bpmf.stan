data {
    int<lower=1> n_users;
    int<lower=1> n_items;

    int<lower=1,upper=min(n_users,n_items)> rank;

    // observed data
    int<lower=1,upper=n_users*n_items> n_obs;
    int<lower=1,upper=n_users> obs_users[n_obs];
    int<lower=1,upper=n_items> obs_items[n_obs];
    real obs_ratings[n_obs];

    // fixed hyperparameters
    real<lower=0> rating_std;

    vector[rank] mu_0; // usually zero
    int<lower=1> nu_0; // usually == rank
    cov_matrix[rank] w_0; // usually identity
}

parameters {
    vector[rank] U[n_users];
    vector[rank] V[n_items];

    vector[rank] mu_u;
    cov_matrix[rank] cov_u;

    vector[rank] mu_v;
    cov_matrix[rank] cov_v;
}

model {
    // observed data likelihood
    for (n in 1:n_obs)
        obs_ratings[n] ~ normal(U[obs_users[n]]' * V[obs_items[n]], rating_std);

    // prior on latent factors
    for (i in 1:n_users)
        U[i] ~ multi_normal(mu_u, cov_u);
    for (j in 1:n_items)
        V[j] ~ multi_normal(mu_v, cov_v);

    // hyperpriors on latent factor hyperparams
    mu_u ~ multi_normal(mu_0, cov_u);
    mu_v ~ multi_normal(mu_0, cov_v);

    cov_u ~ inv_wishart(nu_0, w_0);
    cov_v ~ inv_wishart(nu_0, w_0);
}
