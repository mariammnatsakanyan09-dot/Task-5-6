import os
import json
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

# Set number of threads for CPU
torch.set_num_threads(os.cpu_count())

# Paths
data_path = os.environ.get('SNLS_DATA')
if data_path is None:
    raise EnvironmentError("SNLS_DATA environment variable not set")
print(f"Loading data from {data_path}")

# Load .mat file
mat = scipy.io.loadmat(data_path)
u_raw = mat['V1'].squeeze()  # shape (131072,)
y_raw = mat['V2'].squeeze()

# Excerpt indices
start = 42650
end = 46746  # exclusive
u = u_raw[start:end].astype(np.float32)
y = y_raw[start:end].astype(np.float32)

# Center signals
u = u - np.mean(u)
y = y - np.mean(y)

# Split into train/test
n_train = 3072
u_train = u[:n_train]
y_train = y[:n_train]
u_test = u[n_train:]
y_test = y[n_train:]

# Time step
fs = 610.35
dt = 1.0 / fs

# Compute scaling statistics from training set
mu_y, sigma_y = np.mean(y_train), np.std(y_train) + 1e-8
# velocity via finite difference
v_train = np.gradient(y_train, dt)
mu_v, sigma_v = np.mean(v_train), np.std(v_train) + 1e-8
mu_u, sigma_u = np.mean(u_train), np.std(u_train) + 1e-8

scaling = {
    'mu_y': float(mu_y), 'sigma_y': float(sigma_y),
    'mu_v': float(mu_v), 'sigma_v': float(sigma_v),
    'mu_u': float(mu_u), 'sigma_u': float(sigma_u)
}

# Save scaling
os.makedirs('results', exist_ok=True)
with open('results/scaling.json', 'w') as f:
    json.dump(scaling, f)

# Helper to scale inputs
def scale_inputs(y_val, v_val, u_val):
    y_s = (y_val - mu_y) / sigma_y
    v_s = (v_val - mu_v) / sigma_v
    u_s = (u_val - mu_u) / sigma_u
    return torch.tensor([y_s, v_s, u_s], dtype=torch.float32)

# Define network
class SilverPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)  # output acceleration
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Numerical acceleration via central difference
def central_diff_acceleration(signal, dt):
    # signal: 1D numpy array
    acc = np.zeros_like(signal)
    # central difference for interior
    acc[1:-1] = (signal[2:] - 2*signal[1:-1] + signal[:-2]) / (dt**2)
    # forward difference at start
    acc[0] = (signal[2] - 2*signal[1] + signal[0]) / (dt**2)
    # backward difference at end
    acc[-1] = (signal[-1] - 2*signal[-2] + signal[-3]) / (dt**2)
    return acc.astype(np.float32)

# RK4 integration step with linear interpolation for u at midpoints
def rk4_step(y_n, v_n, u_fn, net, dt):
    # u_fn is a callable that returns u at any time (linear interpolation)
    # k1
    u1 = u_fn(0.0)
    inp1 = scale_inputs(y_n, v_n, u1)
    a1 = net(inp1)
    k1_y = v_n
    k1_v = a1.item()
    # k2
    u2 = u_fn(dt/2)
    y2 = y_n + 0.5*dt*k1_y
    v2 = v_n + 0.5*dt*k1_v
    inp2 = scale_inputs(y2, v2, u2)
    a2 = net(inp2)
    k2_y = v2
    k2_v = a2.item()
    # k3
    u3 = u_fn(dt/2)
    y3 = y_n + 0.5*dt*k2_y
    v3 = v_n + 0.5*dt*k2_v
    inp3 = scale_inputs(y3, v3, u3)
    a3 = net(inp3)
    k3_y = v3
    k3_v = a3.item()
    # k4
    u4 = u_fn(dt)
    y4 = y_n + dt*k3_y
    v4 = v_n + dt*k3_v
    inp4 = scale_inputs(y4, v4, u4)
    a4 = net(inp4)
    k4_y = v4
    k4_v = a4.item()
    # update
    y_next = y_n + (dt/6.0)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_next = v_n + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return y_next, v_next

# Create a callable for u given time (linear interpolation)
def make_u_interpolator(u_array, dt):
    # u_array defined at t = n*dt, n=0..len-1
    def u_at_time(t):
        # t in [0, len*dt)
        n = t / dt
        n0 = int(np.floor(n))
        if n0 < 0:
            return u_array[0]
        if n0 >= len(u_array)-1:
            return u_array[-1]
        alpha = n - n0
        return (1-alpha)*u_array[n0] + alpha*u_array[n0+1]
    return u_at_time

# Training parameters
rollout_len = 256
max_steps = 8000
val_interval = 500
lr = 1e-3

device = torch.device('cpu')
net = SilverPINN().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-4)

# Precompute numerical acceleration for training set (for loss)
acc_train_data = central_diff_acceleration(y_train, dt)

best_train_rmse = float('inf')
best_model_path = 'results/best_model_temp.pt'

# Training loop
for step in range(max_steps):
    # Sample random start index within training region (ensuring rollout fits)
    max_start = n_train - rollout_len
    if max_start <= 0:
        i0 = 0
    else:
        i0 = np.random.randint(0, max_start+1)
    # Extract window
    y_win = y_train[i0:i0+rollout_len]
    u_win = u_train[i0:i0+rollout_len]
    # Compute velocity via finite difference on window (same scheme as used for init)
    v_win = np.gradient(y_win, dt)
    # Numerical acceleration for window
    acc_win_data = acc_train_data[i0:i0+rollout_len]
    
    # Prepare interpolator for u in this window (relative to window start)
    u_interp = make_u_interpolator(u_win, dt)
    
    # Initialize state at start of window
    y_cur = y_win[0]
    v_cur = v_win[0]
    
    # Storage for predicted accelerations at each step (for loss)
    pred_accs = []
    
    # RK4 integration over the window
    for k in range(rollout_len):
        # Predict acceleration at current state
        inp = scale_inputs(y_cur, v_cur, u_win[k])
        a_pred = net(inp)
        pred_accs.append(a_pred)
        # RK4 step to get next state
        y_cur, v_cur = rk4_step(y_cur, v_cur, u_interp, net, dt)
    
    pred_accs_tensor = torch.stack(pred_accs)  # shape (rollout_len,)
    loss = nn.functional.mse_loss(pred_accs_tensor, torch.from_numpy(acc_win_data).to(device))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Validation every val_interval steps
    if (step+1) % val_interval == 0:
        net.eval()
        with torch.no_grad():
            # Free-run on training excerpt
            u_train_interp = make_u_interpolator(u_train, dt)
            y_pred = np.zeros(n_train, dtype=np.float32)
            y_cur = y_train[0]
            v_cur = v_train[0]
            y_pred[0] = y_cur
            for t in range(1, n_train):
                # Predict acceleration
                inp = scale_inputs(y_cur, v_cur, u_train[t-1])
                a_pred = net(inp)
                # RK4 step
                y_cur, v_cur = rk4_step(y_cur, v_cur, lambda tt: u_train_interp(tt), net, dt)
                y_pred[t] = y_cur
            # Compute RMSE
            rmse = np.sqrt(np.mean((y_pred - y_train)**2))
            if rmse < best_train_rmse:
                best_train_rmse = rmse
                torch.save(net.state_dict(), best_model_path)
                print(f"Step {step+1}: new best train RMSE = {rmse:.6f}", flush=True)
        net.train()

# After training, load best model for evaluation
net.load_state_dict(torch.load(best_model_path, map_location=device))
net.eval()

# Free-run on full excerpt (4096 samples)
u_full = np.concatenate([u_train, u_test])
y_full = np.concatenate([y_train, y_test])
v_full = np.gradient(y_full, dt)  # initial velocity for t=0? we'll compute v0 from y[0],y[1]
# Actually we need v0 from finite difference as per spec
v0 = (y_full[1] - y_full[0]) / dt

u_full_interp = make_u_interpolator(u_full, dt)

y_pred_full = np.zeros_like(y_full, dtype=np.float32)
y_cur = y_full[0]
v_cur = v0
y_pred_full[0] = y_cur
for t in range(1, len(y_full)):
    inp = scale_inputs(y_cur, v_cur, u_full[t-1])
    a_pred = net(inp)
    y_cur, v_cur = rk4_step(y_cur, v_cur, lambda tt: u_full_interp(tt), net, dt)
    y_pred_full[t] = y_cur

# Compute RMSE on train and test segments
train_pred = y_pred_full[:n_train]
test_pred = y_pred_full[n_train:]

train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
test_rmse = np.sqrt(np.mean((test_pred - y_test)**2))

# Print exactly two lines
print(f"TRAIN_RMSE={train_rmse}")
print(f"TEST_RMSE={test_rmse}")