import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

# Set number of threads for torch
torch.set_num_threads(os.cpu_count())

# Load data
data_path = os.environ.get('SNLS_DATA')
if not data_path:
    raise EnvironmentError("SNLS_DATA environment variable not set")
mat = sio.loadmat(data_path)
u_raw = mat['V1'].squeeze()
y_raw = mat['V2'].squeeze()

# Excerpt
start = 42650
end = 46746  # exclusive, gives 4096 samples
excerpt = slice(start, end)
u = u_raw[excerpt] - np.mean(u_raw[excerpt])
y = y_raw[excerpt] - np.mean(y_raw[excerpt])

# Sampling frequency
fs = 610.35  # Hz
dt = 1.0 / fs

# Central differences for velocity and acceleration
N = len(u)
y_dot = np.zeros(N)
y_ddot = np.zeros(N)
# Interior points
if N >= 3:
    y_dot[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    y_ddot[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dt ** 2)
    # One-sided at boundaries (copy interior)
    y_dot[0] = y_dot[1]
    y_dot[-1] = y_dot[-2]
    y_ddot[0] = y_ddot[1]
    y_ddot[-1] = y_ddot[-2]
else:
    raise ValueError("Excerpt too short for differentiation")

# Split into training and test
N_train = 3072
u_train = u[:N_train]
y_train = y[:N_train]
y_dot_train = y_dot[:N_train]
y_ddot_train = y_ddot[:N_train]
u_test = u[N_train:]
y_test = y[N_train:]
y_dot_test = y_dot[N_train:]
y_ddot_test = y_ddot[N_train:]

# Normalize inputs (y, y_dot, u) using training statistics
def normalize(x, mean=None, std=None):
    if mean is None:
        mean = np.mean(x)
    if std is None:
        std = np.std(x)
    return (x - mean) / std, mean, std

u_train_n, u_mean, u_std = normalize(u_train)
y_train_n, y_mean, y_std = normalize(y_train)
ydot_train_n, ydot_mean, ydot_std = normalize(y_dot_train)

# Apply same normalization to test set
u_test_n = (u_test - u_mean) / u_std
y_test_n = (y_test - y_mean) / y_std
ydot_test_n = (y_dot_test - ydot_mean) / ydot_std

# Prepare training tensors for one-step prediction
X_train = torch.stack([
    torch.from_numpy(y_train_n).float(),
    torch.from_numpy(ydot_train_n).float(),
    torch.from_numpy(u_train_n).float()
], dim=1)  # shape (N_train, 3)
yddot_target = torch.from_numpy(y_ddot_train).float()

# Define a small fully-connected tanh network
class AccNet(nn.Module):
    def __init__(self, hidden_size=64, num_hidden=2):
        super().__init__()
        self.activation = nn.Tanh()
        self.input_layer = nn.Linear(3, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x).squeeze(-1)

net = AccNet()

# Training: one-step MSE loss
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
net.train()
num_epochs = 20000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred = net(X_train)
    loss = loss_fn(pred, yddot_target)
    loss.backward()
    optimizer.step()
    # Optional: print loss occasionally (commented out to avoid extra output)
    # if epoch % 5000 == 0:
    #     print(f'Epoch {epoch}, loss {loss.item():.6f}')

net.eval()

# Function to compute acceleration from network given raw (unnormalized) y, v, u
def acceleration(y_val, v_val, u_val):
    y_n = (y_val - y_mean) / y_std
    v_n = (v_val - ydot_mean) / ydot_std
    u_n = (u_val - u_mean) / u_std
    inp = torch.tensor([y_n, v_n, u_n], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        acc = net(inp).item()
    return acc

# RK4 step using the network for acceleration
def rk4_step(y_cur, v_cur, u_val):
    # Stage 1
    a1 = acceleration(y_cur, v_cur, u_val)
    k1y = v_cur
    k1v = a1
    # Stage 2
    y2 = y_cur + 0.5 * dt * k1y
    v2 = v_cur + 0.5 * dt * k1v
    a2 = acceleration(y2, v2, u_val)
    k2y = v2
    k2v = a2
    # Stage 3
    y3 = y_cur + 0.5 * dt * k2y
    v3 = v_cur + 0.5 * dt * k2v
    a3 = acceleration(y3, v3, u_val)
    k3y = v3
    k3v = a3
    # Stage 4
    y4 = y_cur + dt * k3y
    v4 = v_cur + dt * k3v
    a4 = acceleration(y4, v4, u_val)
    k4y = v4
    k4v = a4
    # Combine
    y_next = y_cur + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_next = v_cur + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y_next, v_next

# Free-run over the whole excerpt
y_pred = np.zeros_like(y)
v_pred = np.zeros_like(y)
# Initial condition: use first sample of centered y and estimated velocity
y_cur = y[0]
v_cur = y_dot[0]
y_pred[0] = y_cur
v_pred[0] = v_cur
for k in range(len(u) - 1):
    y_cur, v_cur = rk4_step(y_cur, v_cur, u[k])
    y_pred[k + 1] = y_cur
    v_pred[k + 1] = v_cur

# Compute RMSE
train_rmse = np.sqrt(np.mean((y_pred[:N_train] - y[:N_train]) ** 2))
test_rmse = np.sqrt(np.mean((y_pred[N_train:] - y[N_train:]) ** 2))

# Optionally compute LS-derived cubic-to-linear ratio for comment
# Build design matrix for LS: y_ddot = -a*y_dot - b*y - c*y^3 + g*u
X_ls = np.column_stack((-y_dot, -y, -(y**3), u))
theta, _, _, _ = np.linalg.lstsq(X_ls, y_ddot, rcond=None)
a_ls, b_ls, c_ls, g_ls = theta
ratio = np.abs(c_ls) * (0.2 ** 2) / np.abs(b_ls)

# Print comment line (optional) then the two required lines
print(f'# LS |c|y^2/|b| at |y|=0.2 V = {ratio:.6f}')
print(f'TRAIN_RMSE={train_rmse:.6f}')
print(f'TEST_RMSE={test_rmse:.6f}')