import numpy as np
import random
from scipy.stats import multivariate_normal

def simulate_data(n=1200, p_con=75, p_bi=25, p_active_con=15, p_active_bi=5, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    p_total = p_con + p_bi
    rho = 0.75
    sigma = rho ** np.abs(np.subtract.outer(np.arange(p_total),np.arange(p_total)))
    x_0 = multivariate_normal.rvs(mean=np.zeros(p_total), cov=sigma, size=n)
    x_con = x_0[:, :p_con]
    x_bi = (x_0[:, p_con:] > 0).astype(int)
    x = np.hstack([x_con, x_bi])

    con_ind = np.random.choice(p_con, p_active_con, replace=False)
    bi_ind = np.random.choice(np.arange(p_con, p_total), p_active_bi, replace=False)

    b = np.zeros(p_total)
    b[con_ind] = np.random.choice([-1.2, 1.2], p_active_con)
    b[bi_ind] = np.random.choice([-2, 2], p_active_bi)

    h0 = 2.4
    lin_term = np.clip(x @ b, -50, 50)
    h = h0 * np.exp(lin_term)  #hazard score for each sample
    t0 = np.random.exponential(1 / h)  #true survival time
    c = 1.5
    cens = np.random.exponential(1 / c, size=n)  #censoring time
    t = np.minimum(t0, cens) #observation time
    delta = (t0 < cens).astype(bool)  # if observed the event then 1 else 0

    return x, t, delta