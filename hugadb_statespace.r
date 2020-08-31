library(rstan)
library(dplyr)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

fname <- "data/Data/HuGaDB_v1_bicycling_01_01.txt"
datarange <- c(901:1050)
df <- read.csv(fname, sep = '\t', skip = 3)
df <- df %>% select(starts_with("gyro")) %>% slice(datarange)
minx <- min(df)
maxx <- max(df)
df <- 2 / (maxx - minx) * (df - minx) - 1

X <- t(as.matrix(df))
D <- nrow(X)
T <- ncol(X)
K <- 2

X_missing <- X
X_missing[-c(1:8), -c(1:75)] <- 0
X_missing[c(1:12), c(1:60)] <- 0
X_misinds <- matrix(rep(0, D * T), nrow = D, ncol = T)
X_misinds[-c(1:8), -c(1:75)] <- 1
X_misinds[c(1:12), c(1:60)] <- 1

data <- list(D = D, T = T, K = K,
             Y = X_missing,
             Y_misind = X_misinds)
model <- stan_model("./hugadb_statespace.stan")
fit <- sampling(model, data = data, iter = 5000, warmup = 3000)
ext_preds <- rstan::extract(fit, pars = c("Y_new"))
quantile_preds <- array(0, dim = c(3, D, T))
for (d in 1:D) {
    for (t in 1:T) {
        quantile_preds[1, d, t] <- quantile(ext_preds$Y_new[, d, t], 0.025)
        quantile_preds[2, d, t] <- mean(ext_preds$Y_new[, d, t])
        quantile_preds[3, d, t] <- quantile(ext_preds$Y_new[, d, t], 0.975)
    }
}
save(quantile_preds, file = "quantiles_var1.rda")
load("quantiles.rda")

df.0025 <- data.frame(quantile_preds[1, , ])
df.mean <- data.frame(quantile_preds[2, , ])
df.0975 <- data.frame(quantile_preds[3, , ])
write.csv(df.0025, file = "df_0025.csv")
write.csv(df.mean, file = "df_mean.csv")
write.csv(df.0975, file = "df_0975.csv")

