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
  real<lower=0> rating_std; // observation noise std deviation, usually 1/2

  vector[rank] mu_0; // mean for feature means, usually zero

  // feature mean covariances are beta_0 * inv wishart(nu_0, w_0)
  real<lower=0> beta_0; // usually 2
  int<lower=rank> nu_0; // deg of freedom, usually == rank
  cov_matrix[rank] w_0; // scale matrix, usually identity
}

parameters {
  // latent factors
  matrix[n_users, rank] U;
  matrix[n_users, rank] V;

  // means and covs on latent factors
  vector[rank] mu_u;
  vector[rank] mu_v;
  cov_matrix[rank] cov_u;
  cov_matrix[rank] cov_v;
}

transformed parameters {
  matrix[n_users, n_items] predictions;
  predictions <- U * V';
}

model {
  // observed data likelihood
  for (n in 1:n_obs)
    obs_ratings[n] ~ normal(predictions[obs_users[n],obs_items[n]], rating_std);
    // obs_ratings[n] ~ normal(U[obs_users[n]] * V[obs_items[n]]', rating_std);

  // prior on latent factors
  for (i in 1:n_users)
    U[i]' ~ multi_normal(mu_u, cov_u);
  for (j in 1:n_items)
    V[j]' ~ multi_normal(mu_v, cov_v);

  // hyperpriors on latent factor hyperparams
  mu_u ~ multi_normal(mu_0, cov_u / beta_0);
  mu_v ~ multi_normal(mu_0, cov_v / beta_0);
  cov_u ~ inv_wishart(nu_0, w_0);
  cov_v ~ inv_wishart(nu_0, w_0);
}

/*
generated quantities {
  real training_rmse;
  training_rmse <- 0;
  for (i in 1:n_obs) {
    training_rmse <- training_rmse
      + square(predictions[obs_users[i], obs_items[i]] - obs_ratings[i]);
  }
  training_rmse <- sqrt(training_rmse);
}
*/
