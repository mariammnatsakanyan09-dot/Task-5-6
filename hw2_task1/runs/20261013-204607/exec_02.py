import os
import json
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(os.cpu_count())

# Load data
data_path = os.environ.get('SNLS_DATA')
mat = scipy.io.loadmat(data_path)
u_raw = mat['V1'].squeeze()
y_raw = mat['V2'].squeeze()
start = 42650
end = 46746
u = u_raw[start:end].astype(np.float32) - np.mean(u_raw[start:end])
y = y_raw[start:end].astype(np.float32) - np.mean(y_raw[start:end])

n_train = 3072
u_train = u[:n_train]
y_train = y[:n_train]
u_test = u[n_train:]
y_test = y[n_train:]

fs = 610.35
dt = 1.0 / fs

# Scaling from training set
mu_y, sigma_y = np.mean(y_train), np.std(y_train) + 1e-8
v_train = np.gradient(y_train, dt)
mu_v, sigma_v = np.mean(v_train), np.std(v_train) + 1e-8
mu_u, sigma_u = np.mean(u_train), np.std(u_train) + 1e-8

scaling = {
    'mu_y': float(mu_y), 'sigma_y': float(sigma_y),
    'mu_v': float(mu_v), 'sigma_v': float(sigma_v),
    'mu_u': float(mu_u), 'sigma_u': float(sigma_u)
}
os.makedirs('results', exist_ok=True)
with open('results/scaling.json', 'w') as f:
    json.dump(scaling, f)

def scale_inputs(y_val, v_val, u_val):
    y_s = (y_val - mu_y) / sigma_y
    v_s = (v_val - mu_v) / sigma_v
    u_s = (u_val - mu_u) / sigma_u
    return torch.tensor([y_s, v_s, u_s], dtype=torch.float32)

class SilverPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def central_diff_acceleration(signal, dt):
    acc = np.zeros_like(signal)
    acc[1:-1] = (signal[2:] - 2*signal[1:-1] + signal[:-2]) / (dt**2)
    acc[0] = (signal[2] - 2*signal[1] + signal[0]) / (dt**2)
    acc[-1] = (signal[-1] - 2*signal[-2] + signal[-3]) / (dt**2)
    return acc.astype(np.float32)

acc_train_data = central_diff_acceleration(y_train, dt)
print(f"Acceleration data stats: mean={np.mean(acc_train_data):.6f}, std={np.std(acc_train_data):.6f}, max={np.max(np.abs(acc_train_data)):.6f}")

# Network output scaling? Let's see what random network gives
net = SilverPINN()
with torch.no_grad():
    # random input
    inp = torch.randn(3)
    out = net(inp)
    print(f"Random network output example: {out.item():.6f}")

# Training
device = torch.device('cpu')
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)

rollout_len = 64
max_steps = 2000
val_interval = 200

best_train_rmse = float('inf')
best_model_path = 'results/best_model_temp.pt'

for step in range(max_steps):
    # Random start index
    max_start = n_train - rollout_len
    if max_start <= 0:
        i0 = 0
    else:
        i0 = np.random.randint(0, max_start+1)
    y_win = y_train[i0:i0+rollout_len]
    u_win = u_train[i0:i0+rollout_len]
    v_win = np.gradient(y_win, dt)
    acc_win = acc_train_data[i0:i0+rollout_len]
    
    # Convert to tensors
    y_tensor = torch.from_numpy(y_win).to(device)
    v_tensor = torch.from_numpy(v_win).to(device)
    u_tensor = torch.from_numpy(u_win).to(device)
    acc_tensor = torch.from_numpy(acc_win).to(device)
    
    # Predict acceleration for each time step using true y, v, u
    inp_batch = torch.stack([scale_inputs(y_tensor[t], v_tensor[t], u_tensor[t]) for t in range(rollout_len)], dim=0)
    pred_acc = net(inp_batch)  # shape (rollout_len,)
    loss = nn.functional.mse_loss(pred_acc, acc_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if (step+1) % val_interval == 0:
        net.eval()
        with torch.no_grad():
            # Free-run on training excerpt
            def u_interp(t):
                # linear interpolation
                n = t / dt
                n0 = int(np.floor(n))
                if n0 < 0:
                    return u_train[0]
                if n0 >= len(u_train)-1:
                    return u_train[-1]
                alpha = n - n0
                return (1-alpha)*u_train[n0] + alpha*u_train[n0+1]
            y_pred = np.zeros(n_train, dtype=np.float32)
            y_cur = y_train[0]
            v_cur = v_train[0]
            y_pred[0] = y_cur
            for t in range(1, n_train):
                # RK4 step using network acceleration
                u1 = u_interp(0.0)
                inp1 = scale_inputs(y_cur, v_cur, u1)
                a1 = net(inp1)
                k1_y = v_cur
                k1_v = a1.item()
                u2 = u_interp(dt/2)
                y2 = y_cur + 0.5*dt*k1_y
                v2 = v_cur + 0.5*dt*k1_v
                inp2 = scale_inputs(y2, v2, u2)
                a2 = net(inp2)
                k2_y = v2
                k2_v = a2.item()
                u3 = u_interp(dt/2)
                y3 = y_cur + 0.5*dt*k2_y
                v3 = v_cur + 0.5*dt*k2_v
                inp3 = scale_inputs(y3, v3, u3)
                a3 = net(inp3)
                k3_y = v3
                k3_v = a3.item()
                u4 = u_interp(dt)
                y4 = y_cur + dt*k3_y
                v4 = v_cur + dt*k3_v
                inp4 = scale_inputs(y4, v4, u4)
                a4 = net(inp4)
                k4_y = v4
                k4_v = a4.item()
                y_next = y_cur + (dt/6.0)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
                v_next = v_cur + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
                y_cur, v_cur = y_next, v_next
                y_pred[t] = y_cur
            rmse = np.sqrt(np.mean((y_pred - y_train)**2))
            if rmse < best_train_rmse:
                best_train_rmse = rmse
                torch.save(net.state_dict(), best_model_path)
                print(f"Step {step+1}: new best train RMSE = {rmse:.6f}", flush=True)
        net.train()

# Load best model
net.load_state_dict(torch.load(best_model_path, map_location=device))
net.eval()

# Free-run on full excerpt
def u_full_interp(t):
    n = t / dt
    n0 = int(np.floor(n))
    if n0 < 0:
        return u[0]
    if n0 >= len(u)-1:
        return u[-1]
    alpha = n - n0
    return (1-alpha)*u[n0] + alpha*u[n0+1]

y_pred_full = np.zeros_like(y, dtype=np.float32)
y_cur = y[0]
v_cur = (y[1] - y[0]) / dt  # finite difference as per spec
y_pred_full[0] = y_cur
for t in range(1, len(y)):
    # RK4 step
    u1 = u_full_interp(0.0)
    inp1 = scale_inputs(y_cur, v_cur, u1)
    a1 = net(inp1)
    k1_y = v_cur
    k1_v = a1.item()
    u2 = u_full_interp(dt/2)
    y2 = y_cur + 0.5*dt*k1_y
    v2 = v_cur + 0.5*dt*k1_v
    inp2 = scale_inputs(y2, v2, u2)
    a2 = net(inp2)
    k2_y = v2
    k2_v = a2.item()
    u3 = u_full_interp(dt/2)
    y3 = y_cur + 0.5*dt*k2_y
    v3 = v_cur + 0.5*dt*k2_v
    inp3 = scale_inputs(y3, v3, u3)
    a3 = net(inp3)
    k3_y = v3
    k3_v = a3.item()
    u4 = u_full_interp(dt)
    y4 = y_cur + dt*k3_y
    v4 = v_cur + dt*k3_v
    inp4 = scale_inputs(y4, v4, u4)
    a4 = net(inp4)
    k4_y = v4
    k4_v = a4.item()
    y_next = y_cur + (dt/6.0)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_next = v_cur + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    y_cur, v_cur = y_next, v_next
    y_pred_full[t] = y_cur

train_pred = y_pred_full[:n_train]
test_pred = y_pred_full[n_train:]
train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
test_rmse = np.sqrt(np.mean((test_pred - y_test)**2))

print(f"TRAIN_RMSE={train_rmse}")
print(f"TEST_RMSE={test_rmse}")