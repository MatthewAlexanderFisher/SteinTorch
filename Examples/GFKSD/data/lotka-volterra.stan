functions {
  real[] dz_dt(real t,       // time
               real[] z,     // system state {prey, predator}
               real[] theta, // parameters
               real[] x_r,   // unused data
               int[] x_i) {
    real u = z[1];
    real v = z[2];

    real alpha = theta[1];
    real beta = theta[2];
    real gamma = theta[3];
    real delta = theta[4];

    real du_dt = (alpha - beta * v) * u;
    real dv_dt = (-gamma + delta * u) * v;

    return { du_dt, dv_dt };
  }
}
data {
  int<lower = 0> N;          // number of measurement times
  real ts[N];                // measurement times > 0
  real<lower = 0> y[N, 2];   // measured populations
}
parameters {
  real thet[4];   // { alpha, beta, gamma, delta }
  real z_ini[2];  // initial population
  real sigm[2];   // measurement errors
}
transformed parameters {
  real theta[4] = exp(thet);
  real z_init[2] = exp(z_ini);
  real sigma[2] = exp(sigm);
  real z[N, 2]
    = integrate_ode_rk45(dz_dt, z_init, 0, ts, theta,
                         rep_array(0.0, 0), rep_array(0, 0),
                         1e-5, 1e-3, 5e2);
}

model {
  theta[{1, 3}] ~ lognormal(log(0.7), 0.6);
  theta[{2, 4}] ~ lognormal(log(0.02), 0.3);
  z_init ~ lognormal(log(10), 1);
  sigma[1] ~ lognormal(log(0.25), 0.02);
  sigma[2] ~ lognormal(log(0.25), 0.02);
  for (k in 1:2) {
    y[ , k] ~ lognormal(log(z[, k]), sigma[k]);
  }
}

generated quantities {
  real y_init_rep[2];
  real y_rep[N, 2];
  for (k in 1:2) {
    y_init_rep[k] = lognormal_rng(log(z_init[k]), sigma[k]);
    for (n in 1:N)
      y_rep[n, k] = lognormal_rng(log(z[n, k]), sigma[k]);
  }
}
