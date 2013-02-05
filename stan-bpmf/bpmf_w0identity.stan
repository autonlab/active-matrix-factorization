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
}

transformed data {
  real one_over_beta_0;
  matrix[rank, rank] eye;

  for (j in 1:rank) {
    for (i in 1:rank)
      eye[i, j] <- 0.0;
    eye[j, j] <- 1.0;
  }

  one_over_beta_0 <- 1 / beta_0;
}

parameters {
  // latent factors
  matrix[rank, n_users] U; // column-major, so store individual users together
  matrix[rank, n_items] V;

  // means on latent factors; see model sec for details
  vector[rank] mu_u_stdized;
  vector[rank] mu_v_stdized;

  // covariances on latent factors; see model sec for details
  vector<lower=0>[rank] cov_u_c;
  vector[(rank * (rank - 1)) / 2] cov_u_z;
  vector<lower=0>[rank] cov_v_c;
  vector[(rank * (rank - 1)) / 2] cov_v_z;
}

transformed parameters {
  matrix[n_users, n_items] predictions;
  predictions <- U' * V;
}

model {
  vector[rank] mu_u;
  vector[rank] mu_v;

  matrix[rank, rank] cov_u_A;
  matrix[rank, rank] cov_u_L;

  matrix[rank, rank] cov_v_A;
  matrix[rank, rank] cov_v_L;

  int count;

  //////////////////////////////////////////////////////////////////////////////
  // Covariances on the latent factors ~ inv_wishart(nu_0, w_0)

  // The elements of a lower-triangular decomposition of a matrix distributed
  // as wishart(nu_0, I). See section 13.1 of the Stan manual for details
  // (the "multivariate reparameterizations" section).
  for (i in 1:rank) {
    cov_u_c[i] ~ chi_square(nu_0 - i + 1); // diagonals are chi-squared
    cov_v_c[i] ~ chi_square(nu_0 - i + 1);
  }
  cov_u_z ~ normal(0, 1); // lower triangle is standard normal
  cov_v_z ~ normal(0, 1);

  // Build up those lower-triangular matrices from their elements.
  count <- 1;
  for (j in 1:rank) {
    for (i in 1:(j-1)) {
      cov_u_A[i, j] <- 0.0;
      cov_v_A[i, j] <- 0.0;
    }
    cov_u_A[j, j] <- sqrt(cov_u_c[j]);
    cov_v_A[j, j] <- sqrt(cov_v_c[j]);
    for (i in (j+1):rank) {
      cov_u_A[i, j] <- cov_u_z[count];
      cov_v_A[i, j] <- cov_v_z[count];
      count <- count + 1;
    }
  }

  // Find Cholesky-style factors of the covariance matrices.
  cov_u_L <- mdivide_left_tri_low(cov_u_A, eye);
  cov_v_L <- mdivide_left_tri_low(cov_v_A, eye);


  //////////////////////////////////////////////////////////////////////////////
  // Means for the latent factors: multi_normal(mu_0, cov_{u,v} / beta_0)

  // Sample iid normals for efficiency...
  mu_u_stdized ~ normal(0, one_over_beta_0);
  mu_v_stdized ~ normal(0, one_over_beta_0);

  // ...then transform into the desired multivariate normal
  mu_u <- mu_0 + cov_u_L * mu_u_stdized;
  mu_v <- mu_0 + cov_v_L * mu_v_stdized;


  //////////////////////////////////////////////////////////////////////////////
  // The prior on the latent factors we just went to so much trouble to build

  for (i in 1:n_users)
    col(U, i) ~ multi_normal_cholesky(mu_u, cov_u_L);
  for (j in 1:n_items)
    col(V, j) ~ multi_normal_cholesky(mu_v, cov_v_L);


  //////////////////////////////////////////////////////////////////////////////
  // The part that actually uses the data!
  // Assumed to be normal around the predictions by the latent factors.
  for (n in 1:n_obs)
    obs_ratings[n] ~ normal(predictions[obs_users[n],obs_items[n]], rating_std);
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
