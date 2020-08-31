data {
    int D;
    int T;
    int K;
    matrix[D, T] Y;
    int<lower = 0, upper = 1> Y_misind[D, T];
}

transformed data {
    int N_mis;

    N_mis = 0;
    for (d in 1:D) {
        for (t in 1:T) {
            if (Y_misind[d, t] == 1) {
                N_mis += 1;
            }
        }
    }
}

parameters {
    vector[D] mu;
    vector[N_mis] y_mis;
    matrix[D, D] H_1;
    matrix[D, D] H_2;
    real<lower = 0> s_Y1;
    real<lower = 0> s_Y2;
    real<lower = 0> s_Y;
}

model {
    matrix[D, T] Y_com;
    int n;

    n = 1;
    for (d in 1:D) {
        for (t in 1:T) {
            if (Y_misind[d, t] == 0) {
                Y_com[d, t] = Y[d, t];
            } else {
                Y_com[d, t] = y_mis[n];
                n += 1;
            }
        }
    }

    Y_com[:, 1] ~ normal(mu, s_Y);
    Y_com[:, 2] ~ normal(H_1 * Y[:, 1] + mu, s_Y);
    for (t in 3:T) {
        Y_com[:, t] ~ normal(H_1 * Y_com[:, t - 1] + H_2 * Y_com[:, t - 2] + mu, s_Y);
    }
}

generated quantities {
    real Y_new[D, T];
    Y_new[:, 1] = normal_rng(mu, s_Y);
    Y_new[:, 2] = normal_rng(H_1 * to_vector(Y_new[:, 1]) + mu, s_Y);
    for (t in 3:T) {
        Y_new[:, t] = normal_rng(H_1 * to_vector(Y_new[:, t - 1]) +
                                 H_2 * to_vector(Y_new[:, t - 2]) + mu,
                                 s_Y);
    }
}
