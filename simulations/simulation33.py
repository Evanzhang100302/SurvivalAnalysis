import numpy as np
import random
from scipy.stats import multivariate_normal
import itertools

def simulate_data_simu33(n=1200, seed=1):
    p_con, p_bi = 75, 25
    p_active_con, p_active_bi = 15, 5
    p_sqr, p_inter = 10, 10
    p_active_sqr, p_active_inter = 8, 5
    p_total = p_con + p_bi

    np.random.seed(seed)
    random.seed(seed)

    rho = 0.75
    sigma = rho ** np.abs(np.subtract.outer(np.arange(p_total), np.arange(p_total)))
    x_0 = multivariate_normal.rvs(mean=np.zeros(p_total), cov=sigma, size=n)
    x_con = x_0[:, :p_con]
    x_bi = (x_0[:, p_con:] > 0).astype(int)
    x = np.hstack([x_con, x_bi])

    con_ind = np.random.choice(p_con, p_active_con, replace=False)
    bi_ind = np.random.choice(np.arange(p_con, p_total), p_active_bi, replace=False)
    sqr_ind0 = np.random.choice(con_ind, p_sqr, replace=False)
    inter_pairs = random.sample(list(itertools.combinations(con_ind, 2)), p_inter)
    x_sqr = x[:, sqr_ind0] ** 2
    x_inter = np.column_stack([x[:, i] * x[:, j] for i, j in inter_pairs])
    x_full = np.hstack([x, x_sqr, x_inter])

    sqr_indices = np.arange(p_total, p_total + p_sqr)
    inter_indices = np.arange(p_total + p_sqr, p_total + p_sqr + p_inter)
    active_sqr_ind = np.random.choice(sqr_indices, p_active_sqr, replace=False)
    active_inter_ind = np.random.choice(inter_indices, p_active_inter, replace=False)

    b = np.zeros(x_full.shape[1])
    b[con_ind] = np.random.choice([-1.2, 1.2], p_active_con)
    b[bi_ind] = np.random.choice([-2, 2], p_active_bi)
    b[active_sqr_ind] = np.random.choice([-1.2, 1.2], p_active_sqr)
    b[active_inter_ind] = np.random.choice([-2, 2], p_active_inter)

    h0 = 2.4
    lin_term = np.clip(x_full @ b, -50, 50)
    h = h0 * np.exp(lin_term)
    t0 = np.random.exponential(1 / h)
    c = 1.5
    cens = np.random.exponential(1 / c, size=n)
    t = np.minimum(t0, cens)
    delta = (t0 < cens).astype(bool)

    return x_full, t, delta