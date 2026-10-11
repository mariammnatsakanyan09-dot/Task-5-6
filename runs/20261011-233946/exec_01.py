import os
import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

# Set CPU threads
torch.set_num_threads(os.cpu_count())

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load data
mat_path = os.environ.get("SNLS_DATA")
mat_data = scipy.io.loadmat(mat_path)

u_raw = mat_data['V1'].flatten()
y_raw = mat_data['V2'].flatten()

# Extract excerpt: slice [42650 : 46746] (4096 samples)
excerpt_start = 42650
excerpt_end = 46746
u_exc = u_raw[excerpt_start:excerpt_end].copy()
y_exc = y_raw[excerpt_start:excerpt_end].copy()

# Center signals
u_mean = np.mean(u_exc)
y_mean = np.mean(y_exc)
u = u_exc - u_mean
y = y_exc - y_mean

fs = 610.35
dt = 1.0 / fs
N_total = len(y) # 4096
N_train = 3072
N_test = 1024

# Precompute finite difference velocity on training segment
# Centered difference: v[k] = (y[k+1] - y[k-1]) / (2*dt)
# Boundaries: one-sided
v_train = np.zeros(N_train)
v_train[0] = (y[1] - y[0]) / dt
v_train[1:-1] = (y[2:N_train] - y[0:N_train-2]) / (2.0 * dt)
v_train[-1] = (y[N_train-1] - y[N_train-2]) / dt

# Also approximate acceleration for initialization
a_train = np.zeros(N_train)
a_train[1:-1] = (y[2:N_train] - 2*y[1:N_train-1] + y[0:N_train-2]) / (dt**2)
a_train[0] = a_train[1]
a_train[-1] = a_train[-2]

# Convert to torch tensors
u_train_t = torch.tensor(u[:N_train], dtype=torch.float32)
y_train_t = torch.tensor(y[:N_train], dtype=torch.float32)
v_train_t = torch.tensor(v_train, dtype=torch.float32)
a_train_t = torch.tensor(a_train, dtype=torch.float32)

u_all_t = torch.tensor(u, dtype=torch.float32)
y_all_t = torch.tensor(y, dtype=torch.float32)

# 2. Network Architecture
class AccelerationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.y_scale = 0.1
        self.v_scale = 50.0
        self.u_scale = 0.05
        self.out_scale = 40000.0

    def forward(self, y, v, u):
        # inputs can be 1D or 2D tensors of shape (...,)
        y_scaled = y / self.y_scale
        v_scaled = v / self.v_scale
        u_scaled = u / self.u_scale
        
        # stack along last dimension
        inputs = torch.stack([y_scaled, v_scaled, u_scaled], dim=-1)
        out = self.net(inputs)
        return self.out_scale * out.squeeze(-1)

model = AccelerationMLP()

# Phase 0: Quick supervised pretraining on finite-difference acceleration
# to give physics-informed initial weights
optimizer_pre = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
for epoch in range(400):
    optimizer_pre.zero_grad()
    a_pred = model(y_train_t[1:-1], v_train_t[1:-1], u_train_t[1:-1])
    loss_pre = torch.mean((a_pred - a_train_t[1:-1])**2)
    loss_pre.backward()
    optimizer_pre.step()

# RK4 integrator step
def rk4_step(model, y_k, v_k, u_k, u_kp1, dt):
    # k1
    k1_y = v_k
    k1_v = model(y_k, v_k, u_k)
    
    # k2
    u_mid = 0.5 * (u_k + u_kp1)
    y_mid1 = y_k + 0.5 * dt * k1_y
    v_mid1 = v_k + 0.5 * dt * k1_v
    k2_y = v_mid1
    k2_v = model(y_mid1, v_mid1, u_mid)
    
    # k3
    y_mid2 = y_k + 0.5 * dt * k2_y
    v_mid2 = v_k + 0.5 * dt * k2_v
    k3_y = v_mid2
    k3_v = model(y_mid2, v_mid2, u_mid)
    
    # k4
    y_end = y_k + dt * k3_y
    v_end = v_k + dt * k3_v
    k4_y = v_end
    k4_v = model(y_end, v_end, u_kp1)
    
    # next state
    y_next = y_k + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = v_k + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return y_next, v_next

# Rollout function for batch of initial indices
def rollout_batch(model, start_indices, L, u_seq, y_seq, v_seq, dt):
    # start_indices: tensor of shape (B,)
    B = len(start_indices)
    y_curr = y_seq[start_indices]
    v_curr = v_seq[start_indices]
    
    y_preds = []
    y_trues = []
    for step in range(L):
        u_k = u_seq[start_indices + step]
        u_kp1 = u_seq[start_indices + step + 1]
        y_next, v_next = rk4_step(model, y_curr, v_curr, u_k, u_kp1, dt)
        y_preds.append(y_next)
        y_trues.append(y_seq[start_indices + step + 1])
        y_curr, v_curr = y_next, v_next
        
    y_preds = torch.stack(y_preds, dim=1) # (B, L)
    y_trues = torch.stack(y_trues, dim=1) # (B, L)
    return torch.mean((y_preds - y_trues)**2)

# Training curriculum
curriculum = [
    # (steps, L, batch_size, lr)
    (300, 16, 32, 1e-3),
    (300, 32, 16, 5e-4),
    (300, 64, 16, 2e-4),
    (300, 128, 8, 1e-4)
]

start_time = time.time()
for n_steps, L, B, lr in curriculum:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    max_idx = N_train - L - 1
    for step in range(n_steps):
        indices = torch.randint(0, max_idx, (B,))
        optimizer.zero_grad()
        loss = rollout_batch(model, indices, L, u_train_t, y_train_t, v_train_t, dt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

# Free-run evaluation across the full 4096 samples
model.eval()
with torch.no_grad():
    y_hat = np.zeros(N_total)
    v_hat = np.zeros(N_total)
    
    # Starting conditions
    y_hat[0] = y[0]
    v_hat[0] = (y[1] - y[0]) / dt
    
    y_curr = torch.tensor(y_hat[0], dtype=torch.float32)
    v_curr = torch.tensor(v_hat[0], dtype=torch.float32)
    
    for k in range(N_total - 1):
        u_k = u_all_t[k]
        u_kp1 = u_all_t[k + 1]
        y_next, v_next = rk4_step(model, y_curr, v_curr, u_k, u_kp1, dt)
        y_hat[k + 1] = y_next.item()
        v_hat[k + 1] = v_next.item()
        y_curr, v_curr = y_next, v_next

# Compute train and test RMSE
train_rmse = np.sqrt(np.mean((y_hat[:N_train] - y[:N_train])**2))
test_rmse = np.sqrt(np.mean((y_hat[N_train:] - y[N_train:])**2))

print(f"TRAIN_RMSE={train_rmse:.6f}")
print(f"TEST_RMSE={test_rmse:.6f}")
