import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

def run_rsf_simulation(n, n_obs, nsim, seed):
    np.random.seed(seed)
    
    cindex_results = []
    c = 1.5
    p_con = 75
    p_bi = 25
    p_total = p_con + p_bi
    p_active_con = 15
    p_active_bi = 5

    for i in range(nsim*(seed-1)+1, nsim*(seed-1)+nsim+1):
        print(f"Running simulation {i}")

        # Step 1: Covariates
        rho = 0.75
        indices = np.arange(p_total)
        sigma = rho ** np.abs(np.subtract.outer(indices, indices))
        x_0 = multivariate_normal.rvs(mean=np.zeros(p_total), cov=sigma, size=n)

        x_con = x_0[:, :p_con]
        x_bi = (x_0[:, p_con:] > 0).astype(int)
        x = np.hstack([x_con, x_bi])

        con_idx = np.random.choice(range(p_con), size=p_active_con, replace=False)
        bi_idx = np.random.choice(range(p_con, p_total), size=p_active_bi, replace=False)

        con_coef = np.random.choice([-1.2, 1.2], size=p_active_con)
        bi_coef = np.random.choice([-2, 2], size=p_active_bi)

        b0 = np.zeros(p_total)
        b0[con_idx] = con_coef
        b0[bi_idx] = bi_coef

        # Step 2: Hazard & Time
        h0 = 2.4
        h = h0 * np.exp(x @ b0)
        t0 = np.random.exponential(scale=1 / h)
        cens = np.random.exponential(scale=1 / c, size=n)
        t_obs = np.minimum(t0, cens)
        delta = (t0 < cens).astype(bool)

        # Step 3: RSF Fitting
        df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(p_total)])
        df["time"] = t_obs
        df["event"] = delta

        df_train = df.iloc[:n_obs, :]
        df_test = df.iloc[n_obs:, :]

        y_train = Surv.from_dataframe("event", "time", df_train)
        y_test = Surv.from_dataframe("event", "time", df_test)

        X_train = df_train.drop(columns=["time", "event"])
        X_test = df_test.drop(columns=["time", "event"])

        rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, max_features="sqrt", random_state=seed)
        rsf.fit(X_train, y_train)

        surv_preds = rsf.predict_survival_function(X_test, return_array=True)
        # Use expected survival time (area under curve) as predicted risk
        # 获取所有 survival functions（每个样本对应一个 function 对象）
        sf_list = rsf.predict_survival_function(X_test)

# 所有样本的时间点是相同的，取第一个样本的时间向量
        times = sf_list[0].x

# 构建二维矩阵：每行是一个样本的 survival probability 曲线
        surv_preds = np.vstack([sf.y for sf in sf_list])

# 计算每个样本的期望生存时间（AUC）
        expected_time = np.sum(surv_preds * np.diff(np.insert(times, 0, 0)), axis=1)


        cindex = concordance_index_censored(df_test["event"], df_test["time"], -expected_time)[0]
        cindex_results.append({"simulation_time": i, "cindex": cindex})

    return pd.DataFrame(cindex_results)

# Run
df_result = run_rsf_simulation(n=1200, n_obs=1000, nsim=5, seed=1)
df_result.to_csv("rsf_sim_result.csv", index=False)
