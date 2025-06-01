#!/usr/bin/env python
# coding: utf-8

# In[1]:


def run_simulation(max_niter, n, n_obs, dis_threshold, nsim, seed):
    import numpy as np
    import pandas as pd
    import random
    import itertools
    import warnings
    from scipy.stats import multivariate_normal
    from sklearn.neural_network import MLPRegressor
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GridSearchCV
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    pd.options.mode.chained_assignment= None 
    
    c = 1.5  # rate used in generating censored time
    p_con = 75  # number of continuous covariates
    p_bi = 25  # number of binary covariates
    p_total = p_con + p_bi  # total covariates
    p_active_con = 15  # active continuous covariates
    p_active_bi = 5  # active binary covariates
    p_sqr = 10  # number of squared terms to create (from active continuous)
    p_inter = 10  # number of interaction terms to create (from active continuous)
    p_active = p_active_con + p_active_bi  # total active covariates
    p_active_sqr = 8  # active squared terms
    p_active_inter = 5  # active interaction terms
    cindex_result_nn = pd.DataFrame({
        "simulation_time": pd.Series(dtype='int'),
        "cindex": pd.Series(dtype='float')
    })

    
    for i in range(nsim*(seed-1)+1, nsim*(seed-1)+nsim+1):
    
        print(f"Running simulation {i} out of {nsim*seed}")

        np.random.seed(i)
        random.seed(i)

        # AR(1) correlation structure for all covariates:
        rho = 0.75
        indices = np.arange(p_total)
        sigma = rho ** np.abs(np.subtract.outer(indices, indices))

        # Generate p_total correlated covariates for n observations
        x_0 = multivariate_normal.rvs(mean=np.zeros(p_total), cov=sigma, size=n)

        # First p_con columns are continuous
        x_con = x_0[:, :p_con]
        # Next p_bi columns are binary
        x_bi = (x_0[:, p_con:] > 0).astype(int)

        # Combined covariates
        x = np.hstack([x_con, x_bi])

        # Select active covariates randomly:
        con_indice = random.sample(range(p_con), p_active_con)
        bi_indice = random.sample(range(p_con, p_total), p_active_bi)
        active_con = x[:, con_indice]
        active_bi = x[:, bi_indice]

        # --- Create squared and interaction terms ---
        # For squared terms
        sqr_indice0 = random.sample(con_indice, p_sqr)
        x_sqr = x[:, sqr_indice0] ** 2 

        # For interaction terms
        all_pairs = list(itertools.combinations(con_indice, 2))
        inter_pairs = random.sample(all_pairs, p_inter)
        x_inter = np.column_stack([x[:, pair[0]] * x[:, pair[1]] for pair in inter_pairs])

        # Full covariate matrix including squares and interactions
        x_full = np.hstack([x, x_sqr, x_inter])

        # Select active squared and interaction terms from x_full
        # In x_full, the first p_total columns are from x, then p_sqr columns, then p_inter columns.
        # Active squared indices (relative to x_full) are chosen from the squared section:
        sqr_indices = random.sample(range(p_total, p_total + p_sqr), p_active_sqr)
        # Active interaction indices are chosen from the interaction section:
        inter_indices = random.sample(range(p_total + p_sqr, p_total + p_sqr + p_inter), p_active_inter)
        active_sqr = x_full[:, sqr_indices]
        active_inter = x_full[:, inter_indices]

        # Active matrix: combine active continuous, active binary, active squared, active interaction
        x_active = np.hstack([active_con, active_bi, active_sqr, active_inter])

         # --- Coefficients ---
        # Randomly assign coefficients from specified sets
        con_coef = np.random.choice([-1.2, 1.2], size=p_active_con)
        bi_coef = np.random.choice([-2, 2], size=p_active_bi)
        sqr_coef = np.random.choice([-1.2, 1.2], size=p_active_sqr)
        inter_coef = np.random.choice([-2, 2], size=p_active_inter)

         # Initialize full coefficient vector for x_full with zeros
        b0_hr = np.zeros(x_full.shape[1])
        b0_hr[con_indice] = con_coef
        b0_hr[bi_indice] = bi_coef
        b0_hr[sqr_indices] = sqr_coef
        b0_hr[inter_indices] = inter_coef

         # --- Hazard calculation ---
        h0 = 2.4
        # Compute hazard: h = h0 * exp(x_full %*% b0_hr)
        h = h0 * np.exp(np.dot(x_full, b0_hr))

         # --- Generate survival times ---
        t0 = np.random.exponential(scale=1 / h)
        cens = np.random.exponential(scale=1 / c, size=n)
        # tau is computed but not used later; if needed, compute quantile from a sample of 50000 exponential times.
        # For simplicity, we can skip or compute using one typical hazard value.
        # t is the observed time: the minimum of t0 and cens.
        t_obs = np.minimum(t0, cens)
        delta = (t0 < cens).astype(int)  # event indicator: 1 if event occurred, 0 if censored
        lt = np.log(t_obs)
        ids = np.arange(1, n + 1)

        dat = pd.DataFrame(
        columns=["id"] + [f"x{i + 1}" for i in range(x.shape[1])] + ["t", "lt", "delta"])

        dat_full = pd.DataFrame(
        columns=["id"] + [f"x{i + 1}" for i in range(x_full.shape[1])] + ["t", "lt", "delta"])

        dat_active = pd.DataFrame(
        columns=["id"] + [f"x{i + 1}" for i in range(x_active.shape[1])] + ["t", "lt", "delta"])

        # --- Create DataFrames ---
        # dat: includes original covariates (x) with survival info
        dat = pd.concat([
            pd.Series(ids, name="id").astype(int),
            pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(x.shape[1])]),
            pd.Series(t_obs, name="t"),
            pd.Series(lt, name="lt"),
            pd.Series(delta, name="delta")
        ], axis=1)

        # dat_full: includes x_full (covariates with squares and interactions)
        dat_full = pd.concat([
            pd.Series(ids, name="id").astype(int),
            pd.DataFrame(x_full, columns=[f"x{i + 1}" for i in range(x_full.shape[1])]),
            pd.Series(t_obs, name="t"),
            pd.Series(lt, name="lt"),
            pd.Series(delta, name="delta")
        ], axis=1)

        # dat_active: includes only active terms (x_active)
        dat_active = pd.concat([
            pd.Series(ids, name="id").astype(int),
            pd.DataFrame(x_active, columns=[f"x{i + 1}" for i in range(x_active.shape[1])]),
            pd.Series(t_obs, name="t"),
            pd.Series(lt, name="lt"),
            pd.Series(delta, name="delta")
        ], axis=1)

        euclidean_dist_nn_i = []
        nn_cindex_i = []

        dat5 = pd.DataFrame(columns=dat.columns)
        dat5_test = pd.DataFrame(columns=dat.columns)

        dat5 = dat.iloc[0:n_obs, :]
        lt_original = dat.iloc[0:n_obs]['lt']
        dat5_test = dat.iloc[n_obs:n, :]

        niter5 = 0

        while True:
            niter5 += 1

            # Save the current 'lt' column for comparison later.
            lt_old = dat5['lt'].copy()

            # --- Prepare data for model fitting ---
            exclude_vars = ['id', 't', 'delta', 'fitted', 'res']
            cols = [col for col in dat5.columns if col not in exclude_vars]
            nn_data = dat5[cols]

            # Split predictors and target.
            X_train = nn_data.drop(columns=['lt'], errors='ignore')
            y_train = nn_data['lt']

            # --- Fit the model using MLPRegressor instead of lm() ---          
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 50)],
                'alpha': [0.0001, 0.001],
            }

            best_model = MLPRegressor(
                activation='relu',
                learning_rate_init=0.005,
                max_iter=500,
                early_stopping =True,
                learning_rate='adaptive',
                random_state=123
            )

            grid = GridSearchCV(best_model, param_grid, cv=5, verbose=0)
            grid.fit(X_train, y_train)

            final_model = grid.best_estimator_

            # Save the fitted values back into dat2.
            dat5['fitted'] = final_model.predict(X_train)

            # --- Prediction on testing dataset ---
            # Exclude the columns "id", "t", "delta", "lt" from test data.
            test_exclude = ['id', 't', 'delta', 'lt','predict']
            X_test = dat5_test.drop(columns=test_exclude, errors='ignore')
            dat5_test['predict'] = final_model.predict(X_test)

            # --- Compute concordance index using lifelines ---
            cindex_lifelines = concordance_index(dat5_test['lt'], dat5_test['predict'], dat5_test['delta'])
            nn_cindex_i.append(cindex_lifelines)

            # --- Update residuals and adjust log-time for censored observations ---
            dat5['res'] = lt_original - dat5['fitted']

            # Fit the Kaplan-Meier estimator on the residuals.
            kmf = KaplanMeierFitter()
            kmf.fit(durations=dat5['res'], event_observed=dat5['delta'])

            # Order dat5 by residuals.
            dat5.sort_values(by='res', inplace=True)
            dat5.reset_index(drop=True, inplace=True)

            # Get residuals corresponding to censored observations (delta == 0).
            res_cens = dat5.loc[dat5['delta'] == 0, 'res'].values

            # Extract times and survival probabilities from the KM estimator.
            # kmf.survival_function_ is a DataFrame with index = times and column "KM_estimate".
            times = kmf.survival_function_.index.values
            surv = kmf.survival_function_['KM_estimate'].values

            # Create a shifted survival probability vector:
            # In R: temp = c(1, surv[-length(surv)])
            temp = np.concatenate(([1], surv[:-1]))
            # Compute jump changes in survival probability.
            surv_change_event = temp - surv

            # For each censored residual, compute the denominator and numerator for adjustment.
            denom = np.array([kmf.predict(x) for x in res_cens])
            numer = []

            for x in res_cens:
                mask = times >= x
                if np.any(mask):
                    # Sum over (time * jump in survival) for times >= x
                    numer_val = np.sum(times[mask] * surv_change_event[mask])
                else:
                    numer_val = 0
                numer.append(numer_val)
            numer = np.array(numer)

            # Update log-time 'lt' for censored observations:
            mask_cens = dat5['delta'] == 0
            # Make sure division is safe (denom not zero)
            adjustment = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)
            dat5.loc[mask_cens, 'lt'] = dat5.loc[mask_cens, 'fitted'] + adjustment

            # Reorder dat5 by 'id' (if necessary)
            dat5.sort_values(by='id', inplace=True)
            dat5.reset_index(drop=True, inplace=True)

            # --- Compute Euclidean distance for censored observations between updated and previous lt ---
            # Use only rows where delta == 0.
            diff = dat5.loc[dat5['delta'] == 0, 'lt'] - lt_old[dat5['delta'] == 0]
            euclidean_dist = (1 / len(diff)) * np.sqrt(np.sum(diff**2)) if n_obs > 0 else 0
            euclidean_dist_nn_i.append(euclidean_dist)

             # --- Convergence check ---
            if euclidean_dist < dis_threshold:
                print(f"Converged at iteration: {niter5} with max change: {euclidean_dist}")
                break

            if niter5 >= max_niter:
                print(f"nn reached max iterations ({max_niter}), stopping with Euclidean distance: {euclidean_dist}")

                break

            # Optional: print progress every 10 iterations
            if niter5 % 10 == 0:
                print(f"Iteration: {niter5}")

        # After the while loop finishes (i.e. after break)
        new_row_nn = pd.DataFrame({
            "simulation_time": [i],
            "cindex": [nn_cindex_i[niter5 - 1]]
        })
        cindex_result_nn = pd.concat([cindex_result_nn, new_row_nn], ignore_index=True)

    filename = f"cindex_result_nn33_seed_{seed}.csv"
    cindex_result_nn.to_csv(filename, index=False)

    result = f"Simulation complete with seed {seed}"
    return result


# In[ ]:
if __name__ == "__main__":
    # 参数设置
    max_niter = 50
    n = 1200
    n_obs = 1000
    dis_threshold = 1e-4
    nsim = 5
    seed = 1

    # 执行函数
    run_simulation(max_niter, n, n_obs, dis_threshold, nsim, seed)


