import os
import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn

torch.set_num_threads(os.cpu_count() or 4)

# 1. Load data
mat_path = os.environ.get('SNLS_DATA', '/Users/mari/Desktop/Task5-6/SNLS80mV.mat')
mat = scipy.io.loadmat(mat_path)
u_raw = mat['V1'].squeeze()
y_raw = mat['V2'].squeeze()

# Slice [42650 : 46746] (4096 samples)
u_slice = u_raw[42650:46746].astype(np.float32)
y_slice = y_raw[42650:46746].astype(np.float32)

# Subtract means
u_slice -= np.mean(u_slice)
y_slice -= np.mean(y_slice)

dt = 1.0 / 610.35

# Tensors
u_all = torch.tensor(u_slice, dtype=torch.float32)
y_all = torch.tensor(y_slice, dtype=torch.float32)

u_train = u_all[:3072]
y_train = y_all[:3072]

# Compute velocities using central differences
v_train = np.zeros(3072, dtype=np.float32)
v_train[1:-1] = (y_slice[2:3072] - y_slice[:3070]) / (2.0 * dt)
v_train[0] = (y_slice[1] - y_slice[0]) / dt
v_train[-1] = (y_slice[3071] - y_slice[3070]) / dt
v_train = torch.tensor(v_train, dtype=torch.float32)

# 2. Define Model
class AccelerationMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Small initialization on the final layer to prevent explosive gradients early on
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.scale_out = 50000.0

    def forward(self, y, v, u):
        # Scale inputs: y*10, v*0.02, u*10
        # Inputs shape: (B,)
        inp = torch.stack([y * 10.0, v * 0.02, u * 10.0], dim=-1)
        out = self.net(inp).squeeze(-1)
        return out * self.scale_out

model = AccelerationMLP(hidden_dim=64)

# 3. Batched RK4 step
def rk4_step(y, v, u_curr, u_next, dt, net):
    u_mid = 0.5 * (u_curr + u_next)
    
    k1_y = v
    k1_v = net(y, v, u_curr)
    
    k2_y = v + 0.5 * dt * k1_v
    k2_v = net(y + 0.5 * dt * k1_y, k2_y, u_mid)
    
    k3_y = v + 0.5 * dt * k2_v
    k3_v = net(y + 0.5 * dt * k2_y, k3_y, u_mid)
    
    k4_y = v + dt * k3_v
    k4_v = net(y + dt * k3_y, k4_y, u_next)
    
    y_next = y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return y_next, v_next

# 4. Training Curriculum
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

stages = [
    (400, 32, 32, 1e-3),   # (steps, rollout_len, batch_size, lr)
    (400, 64, 16, 5e-4),
    (400, 128, 8, 3e-4),
    (300, 256, 4, 1e-4)
]

N_train = 3072

for stage_idx, (num_steps, L, B, lr) in enumerate(stages):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Sample random starting indices
        idx = torch.randint(0, N_train - L, (B,))
        
        y_curr = y_train[idx]
        v_curr = v_train[idx]
        
        loss = 0.0
        for t in range(L):
            u_c = u_train[idx + t]
            u_n = u_train[idx + t + 1]
            y_next, v_next = rk4_step(y_curr, v_curr, u_c, u_n, dt, model)
            y_target = y_train[idx + t + 1]
            loss = loss + torch.mean((y_next - y_target) ** 2)
            y_curr, v_curr = y_next, v_next
            
        loss = loss / L
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# 5. Free-run evaluation across entire 4096 samples
model.eval()
with torch.no_grad():
    y_pred = torch.zeros(4096, dtype=torch.float32)
    y_curr = y_all[0].clone()
    v_curr = (y_all[1] - y_all[0]) / dt
    y_pred[0] = y_curr
    
    for k in range(4095):
        y_curr, v_curr = rk4_step(y_curr.unsqueeze(0), v_curr.unsqueeze(0), 
                                  u_all[k].unsqueeze(0), u_all[k+1].unsqueeze(0), 
                                  dt, model)
        y_curr = y_curr.squeeze(0)
        v_curr = v_curr.squeeze(0)
        y_pred[k+1] = y_curr

train_rmse = torch.sqrt(torch.mean((y_pred[:3072] - y_all[:3072]) ** 2)).item()
test_rmse = torch.sqrt(torch.mean((y_pred[3072:] - y_all[3072:]) ** 2)).item()

print(f"TRAIN_RMSE={train_rmse:.6f}")
print(f"TEST_RMSE={test_rmse:.6f}")
