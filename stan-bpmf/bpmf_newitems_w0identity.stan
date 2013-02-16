data {
  int<lower=1> n_users;
  int<lower=1> n_fixed_items;
  int<lower=1> n_new_items;

  int<lower=1,upper=min(n_users,n_fixed_items)> rank;

  // observed data - only for the "new" items
  int<lower=1,upper=n_users*n_new_items> n_obs;
  int<lower=1,upper=n_users> obs_users[n_obs];
  int<lower=1,upper=n_new_items> obs_items[n_obs];
  real obs_ratings[n_obs];

  // fixed latent factors for users and "old" items
  vector[rank] U[n_users];
  vector[rank] V_fixed[n_fixed_items];

  // fixed hyperparameters
  real<lower=0> rating_std; // observation noise std deviation, usually 1/2

  vector[rank] mu_0; // mean for feature means, usually zero

  // feature mean covariances are beta_0 * inv wishart(nu_0, eye)
  real<lower=0> beta_0; // usually 2
  int<lower=rank> nu_0; // deg of freedom, usually == rank
}

transformed data {
  real one_over_beta_0;
  vector[rank] nu_0_minus_i;
  matrix[rank, rank] eye;

  for (j in 1:rank) {
    for (i in 1:rank)
      eye[i, j] <- 0.0;
    eye[j, j] <- 1.0;
  }

  one_over_beta_0 <- 1 / beta_0;

  for (i in 1:rank) {
    nu_0_minus_i[i] <- nu_0 - i + 1;
  }
}

parameters {
  // latent factors
  vector[rank] V_new[n_new_items];

  // means on latent factors
  vector[rank] mu_v_stdized;

  // covariances on latent factors; see model sec for details
  vector<lower=0>[rank] cov_v_c;
  vector[(rank * (rank - 1)) / 2] cov_v_z;
}

model {
  vector[rank] mu_v;

  matrix[rank, rank] cov_v_A;
  matrix[rank, rank] cov_v_L;

  int count;

  //////////////////////////////////////////////////////////////////////////////
  // Covariances on the latent factors ~ inv_wishart(nu_0, eye)

  // The elements of a lower-triangular decomposition of a matrix distributed
  // as wishart(nu_0, I). See section 13.1 of the Stan manual for details
  // (the "multivariate reparameterizations" section).
  cov_v_c ~ chi_square(nu_0_minus_i);// diagonals are chi-squared
  cov_v_z ~ normal(0, 1); // lower triangle is standard normal

  // Build up those lower-triangular matrices from their elements.
  count <- 1;
  for (j in 1:rank) {
    for (i in 1:(j-1)) {
      cov_v_A[i, j] <- 0.0;
    }
    cov_v_A[j, j] <- sqrt(cov_v_c[j]);
    for (i in (j+1):rank) {
      cov_v_A[i, j] <- cov_v_z[count];
      count <- count + 1;
    }
  }

  // Find Cholesky-style factors of the covariance matrices.
  cov_v_L <- mdivide_left_tri_low(cov_v_A, eye);


  //////////////////////////////////////////////////////////////////////////////
  // Means for the latent factors: multi_normal(mu_0, cov_{u,v} / beta_0)

  // Sample iid normals for efficiency...
  mu_v_stdized ~ normal(0, one_over_beta_0);

  // ...then transform into the desired multivariate normal
  mu_v <- mu_0 + cov_v_L * mu_v_stdized;


  //////////////////////////////////////////////////////////////////////////////
  // The prior on the latent factors we just went to so much trouble to build

  for (j in 1:n_fixed_items)
    V_fixed[j] ~ multi_normal_cholesky(mu_v, cov_v_L);
  for (j in 1:n_new_items)
    V_new[j] ~ multi_normal_cholesky(mu_v, cov_v_L);


  //////////////////////////////////////////////////////////////////////////////
  // The part that actually uses the data!
  // Assumed to be normal around the predictions by the latent factors.
  {
    vector[n_obs] obs_means;
    for (n in 1:n_obs) {
      obs_means[n] <- dot_product(U[obs_users[n]], V_new[obs_items[n]]);
    }
    obs_ratings ~ normal(obs_means, rating_std);
  }
}
