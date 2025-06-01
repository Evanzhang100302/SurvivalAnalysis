# In[1]:
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_absolute_error
from scipy.stats import multivariate_normal
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class DeepSurv(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim,100)
        self.fc2 = nn.Linear(100,100)
        self.output = nn.Linear(100,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.output(x).squeeze(-1)
    
def cox_ph_loss(risk_scores, durations, events):
    order = torch.argsort(durations, descending=True)
    durations = durations[order]
    events = events[order]
    risk_scores = risk_scores[order]
    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)
    diff = risk_scores - log_cumsum
    loss = -torch.sum(diff * events) / events.sum()
    return loss

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

def train_deepsurv(x_train, t_train, e_train, input_dim, lr=0.01, epochs=100, batch_size=None, verbose=True):
    model = DeepSurv(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

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
    cindex = concordance_index_censored(e_test.astype(bool), t_test, risk_scores)[0]

    risk_order = np.argsort(-risk_scores)
    t_test_sorted = np.sort(t_test)
    t_pred = np.zeros_like(t_test)
    t_pred[risk_order] = t_test_sorted

    mae = mean_absolute_error(t_pred, t_test)
    return cindex, mae

if __name__ == "__main__":
    x, t, delta = simulate_data()
    x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(x, t, delta, test_size=0.2, random_state=42)

    model = train_deepsurv(x_train, t_train, e_train, input_dim=x.shape[1])
    cindex, mae = evaluate_model(model, x_test, t_test, e_test)

    print(f"DeepSurv simu20 C-index: {cindex:.4f}|MAE:{mae:.4f}")

