# Simu20LinearRfcov100 replication in Python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ========== Parameters ==========
max_iter = 10
n = 1200
n_obs = 1000
p_con, p_bi = 75, 25
p_total = p_con + p_bi
rho = 0.75
h0 = 2.4
learning_rate = 0.01
dis_threshold = 1e-4
nsim = 5  # number of simulations per seed
seed_list = [1, 2, 3]  # multiple seeds

# ========== Main Loop for Repeated Simulations ==========

results = {
    "true": [],
    "under": [],
    "over": [],
    "rf": [],
    "nn": []
}

for seed in seed_list:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"==== Seed {seed} ====")

    for sim in range(nsim):
        print(f"-- Simulation {sim+1}/{nsim} --")

        # AR(1) Covariance Matrix
        indices = np.arange(p_total)
        sigma = rho ** np.abs(np.subtract.outer(indices, indices))

        # Generate covariates
        x0 = np.random.multivariate_normal(mean=np.zeros(p_total), cov=sigma, size=n)
        x_con = x0[:, :p_con]
        x_bi = (x0[:, p_con:] > 0).astype(int)
        x = np.hstack([x_con, x_bi])

        # Feature index setup
        con_indices = np.random.choice(p_con, size=15, replace=False)
        bi_indices = np.random.choice(np.arange(p_con, p_total), size=5, replace=False)
        active_indices = np.concatenate([con_indices, bi_indices])
        subset_active = np.random.choice(active_indices, size=5, replace=False)
        inactive_indices = np.setdiff1d(np.arange(p_total), active_indices)

        X_active = x[:, active_indices]
        X_under = np.hstack([x[:, subset_active], x[:, inactive_indices]])
        X_over = x.copy()

        # Coefficients
        beta = np.zeros(p_total)
        beta[con_indices] = np.random.choice([-1.2, 1.2], size=len(con_indices), replace=True)
        beta[bi_indices] = np.random.choice([-2.0, 2.0], size=len(bi_indices), replace=True)

        # Survival times
        hazards = h0 * np.exp(x.dot(beta))
        t0 = -np.log(np.random.rand(n)) / hazards
        cens = np.random.exponential(scale=1/1.5, size=n)
        t = np.minimum(t0, cens)
        delta = (t0 < cens).astype(int)
        lt = np.log(t)

        # Split
        train_idx = np.arange(n_obs)
        test_idx = np.arange(n_obs, n)

        X_dict = {
            'true': X_active,
            'under': X_under,
            'over': X_over,
            'rf': x,
            'nn': x
        }

        for key in X_dict:
            X_train = X_dict[key][train_idx]
            X_test = X_dict[key][test_idx]
            lt_train = lt[train_idx].copy()
            lt_test = lt[test_idx]
            delta_train = delta[train_idx]
            delta_test = delta[test_idx]
            lt_original_train = lt_train.copy()

            converged = False
            n_iter = 0
            while not converged and n_iter < max_iter:
                n_iter += 1
                lt_old = lt_train.copy()

                if key == 'rf':
                    model = RandomForestRegressor(n_estimators=300, max_features=round(p_total/3), min_samples_leaf=5, random_state=seed)
                elif key == 'nn':
                    model = nn.Sequential(
                        nn.Linear(p_total, 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()
                    X_tensor = torch.tensor(X_train, dtype=torch.float32)
                    y_tensor = torch.tensor(lt_train, dtype=torch.float32).view(-1, 1)
                    model.train()
                    for epoch in range(100):
                        optimizer.zero_grad()
                        output = model(X_tensor)
                        loss = criterion(output, y_tensor)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    preds_train = model(X_tensor).detach().numpy().flatten()
                    preds_test = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()
                else:
                    model = LinearRegression().fit(X_train, lt_train)
                    preds_train = model.predict(X_train)
                    preds_test = model.predict(X_test)

                residuals = lt_original_train - preds_train
                kmf = KaplanMeierFitter().fit(residuals, event_observed=delta_train)   #update residuals

                if (delta_train == 0).any():
                    surv = kmf.survival_function_['KM_estimate'].values
                    times = kmf.survival_function_.index.values
                    event_times = []
                    drops = []
                    for i in range(1, len(surv)):
                        if surv[i] < surv[i-1]:
                            event_times.append(times[i])
                            drops.append(surv[i-1] - surv[i])
                    event_times = np.array(event_times)
                    drops = np.array(drops)
                    for j, r in enumerate(residuals):
                        if delta_train[j] == 0:
                            S_r = kmf.predict(r)
                            idx = np.searchsorted(event_times, r, side='left')
                            numer = (event_times[idx:] * drops[idx:]).sum() if idx < len(event_times) else 0.0
                            cond_exp = numer / (S_r if S_r > 1e-8 else 1e-8)
                            lt_train[j] = preds_train[j] + cond_exp

                diff = lt_train[delta_train == 0] - lt_old[delta_train == 0]
                eucl_dist = np.sqrt((diff**2).sum()) / max(1, (delta_train == 0).sum())
                if eucl_dist < dis_threshold:
                    converged = True

            c_index = concordance_index(np.exp(lt_test), preds_test, delta_test)
            results[key].append(c_index)

# Final output
print("\nSummary of Results (Average C-index):")
for k in results:
    print(f"{k}: {np.mean(results[k]):.4f} ± {np.std(results[k]):.4f}")

