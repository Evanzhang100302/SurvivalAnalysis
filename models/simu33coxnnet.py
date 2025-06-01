import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from scipy.stats import multivariate_normal
import itertools
import random
from sklearn.metrics import mean_absolute_error

class CoxNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, dropout=0.3):
        super(CoxNNet, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        return self.output(x).squeeze(-1)

def cox_ph_loss(risk_scores, durations, events):
    order = torch.argsort(durations, descending=True)
    durations, events, risk_scores = durations[order], events[order], risk_scores[order]
    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)
    loss = -torch.sum((risk_scores - log_cumsum) * events) / events.sum()
    return loss

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

def train_coxnnet(x_train, t_train, e_train, input_dim, epochs=100, lr=0.01):
    model = CoxNNet(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    e_train = torch.tensor(e_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = cox_ph_loss(pred, t_train, e_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, x_test, t_test, e_test):
    model.eval()
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        risk_scores = model(x_test_tensor).numpy()
    cindex = concordance_index_censored(e_test, t_test, risk_scores)[0]

    risk_order = np.argsort(-risk_scores)
    t_test_sorted = np.sort(t_test)
    t_pred = np.zeros_like(t_test)
    t_pred[risk_order] = t_test_sorted

    mae = mean_absolute_error(t_pred, t_test)

    return cindex, mae

if __name__ == "__main__":
    x,t,delta = simulate_data_simu33()
    x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(x, t, delta, test_size=0.2, random_state=42)

    model = train_coxnnet(x_train, t_train, e_train, input_dim=x.shape[1])
    cindex, mae = evaluate_model(model, x_test, t_test, e_test)

    print(f"C_index:{cindex:.4f}| MAE:{mae:.4f}")
