import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from itertools import combinations
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def simu33_rfsurv(n=1200, n_obs=1000, nsim=5, seed=1):
    np.random.seed(seed)
    results = []

    # parameters
    c = 1.5
    p_con = 75
    p_bi = 25
    p_total = p_con + p_bi
    p_active_con = 15
    p_active_bi = 5
    p_sqr = 10
    p_inter = 10
    p_active_sqr = 8
    p_active_inter = 5

    for sim in range(nsim * (seed - 1) + 1, nsim * seed + 1):
        print(f"Running simulation {sim}")

        # Step 1: Generate covariates
        sigma = 0.75 ** np.abs(np.subtract.outer(np.arange(p_total), np.arange(p_total)))
        x_0 = multivariate_normal.rvs(mean=np.zeros(p_total), cov=sigma, size=n)
        x_con = x_0[:, :p_con]
        x_bi = (x_0[:, p_con:] > 0).astype(int)
        x = np.hstack([x_con, x_bi])

        # Select active indices
        con_idx = np.random.choice(p_con, p_active_con, replace=False)
        bi_idx = np.random.choice(np.arange(p_con, p_total), p_active_bi, replace=False)

        # Squared and interaction terms
        sqr_idx0 = np.random.choice(con_idx, p_sqr, replace=False)
        x_sqr = x[:, sqr_idx0] ** 2

        inter_candidates = list(combinations(con_idx, 2))
        inter_pairs = np.array(inter_candidates)[np.random.choice(len(inter_candidates), p_inter, replace=False)]
        x_inter = np.column_stack([x[:, i] * x[:, j] for i, j in inter_pairs])

        x_full = np.hstack([x, x_sqr, x_inter])

        # Select active sqr & inter indices
        sqr_idx = np.random.choice(np.arange(p_total, p_total + p_sqr), p_active_sqr, replace=False)
        inter_idx = np.random.choice(np.arange(p_total + p_sqr, p_total + p_sqr + p_inter), p_active_inter, replace=False)

        # Coefficients
        b = np.zeros(x_full.shape[1])
        b[con_idx] = np.random.choice([-1.2, 1.2], p_active_con)
        b[bi_idx] = np.random.choice([-2, 2], p_active_bi)
        b[sqr_idx] = np.random.choice([-1.2, 1.2], p_active_sqr)
        b[inter_idx] = np.random.choice([-2, 2], p_active_inter)

        # Hazard and survival times
        h0 = 2.4
        lin_term = np.clip(x_full @ b, -50, 50)
        h = h0 * np.exp(lin_term)
        t0 = np.random.exponential(1 / h)
        cens = np.random.exponential(1 / c, size=n)
        t = np.minimum(t0, cens)
        delta = (t0 < cens).astype(bool)
        lt = np.log(t)

        # Data for RSF
        X = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(p_total)])
        Y = pd.DataFrame({
            "time": t,
            "event": delta
        })

        # Train/test split
        X_train, X_test = X.iloc[:n_obs], X.iloc[n_obs:]
        y_train = Surv.from_arrays(Y.iloc[:n_obs]["event"], Y.iloc[:n_obs]["time"])
        y_test = Surv.from_arrays(Y.iloc[n_obs:]["event"], Y.iloc[n_obs:]["time"])

        # Fit RSF
        rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, max_features="sqrt", random_state=seed)
        rsf.fit(X_train, y_train)

        # Predict survival function and compute expected time
        sf = rsf.predict_survival_function(X_test)
        times = sf[0].x
        surv_probs = np.vstack([s.y for s in sf])
        expected_time = np.sum(surv_probs * np.diff(np.insert(times, 0, 0)), axis=1)

        # Compute C-index
        cindex = concordance_index_censored(Y.iloc[n_obs:]["event"].values,
                                            Y.iloc[n_obs:]["time"].values,
                                            -expected_time)[0]
        results.append({"simulation_time": sim, "cindex": cindex})

    df_result = pd.DataFrame(results)
    df_result.to_csv(f"33cindex_rfsurv_seed{seed}.csv", index=False)
    return df_result

df_result = simu33_rfsurv(n=1200, n_obs=1000, nsim=5, seed=1)
df_result.to_csv("rsf_sim33_result.csv", index=False)
