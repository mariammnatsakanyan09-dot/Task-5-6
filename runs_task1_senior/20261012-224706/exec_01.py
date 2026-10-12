import os
import sys
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

def load_data():
    data_path = os.environ.get('SNLS_DATA', 'SNLS80mV.mat')
    if not os.path.exists(data_path):
        # search in current directory or parent
        for p in ['SNLS80mV.mat', '../SNLS80mV.mat']:
            if os.path.exists(p):
                data_path = p
                break
    print(f"Loading data from: {data_path}")
    mat = scipy.io.loadmat(data_path)
    print(f"Keys in mat: {[k for k in mat.keys() if not k.startswith('__')]}")
    
    # Silverbox data keys
    if 'V1' in mat and 'V2' in mat:
        u = mat['V1'].squeeze().astype(np.float32)
        y = mat['V2'].squeeze().astype(np.float32)
    elif 'u' in mat and 'y' in mat:
        u = mat['u'].squeeze().astype(np.float32)
        y = mat['y'].squeeze().astype(np.float32)
    else:
        # Find numeric arrays
        arrays = [(k, v.squeeze().astype(np.float32)) for k, v in mat.items() if not k.startswith('__') and isinstance(v, np.ndarray)]
        arrays.sort(key=lambda x: x[1].size, reverse=True)
        u = arrays[0][1]
        y = arrays[1][1]
        print(f"Using keys: {arrays[0][0]} for u, {arrays[1][0]} for y")
        
    print(f"Signal lengths: u={len(u)}, y={len(y)}")
    return u, y

class VectorField(nn.Module):
    def __init__(self, nx=2, hidden_dim=32):
        super().__init__()
        self.nx = nx
        # Linear dynamic component (initialized stable)
        self.A = nn.Parameter(torch.randn(nx, nx) * 0.05)
        self.B = nn.Parameter(torch.randn(nx, 1) * 0.05)
        
        # Nonlinear MLP
        self.mlp = nn.Sequential(
            nn.Linear(nx + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, nx)
        )
        # Initialize MLP output close to zero so linear model guides early training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, u):
        # x: (batch, nx), u: (batch, 1)
        lin = torch.matmul(x, self.A.t()) + torch.matmul(u, self.B.t())
        nonlin = self.mlp(torch.cat([x, u], dim=-1))
        return lin + nonlin

class NeuralODE(nn.Module):
    def __init__(self, nx=2, hidden_dim=32):
        super().__init__()
        self.nx = nx
        self.vf = VectorField(nx, hidden_dim)
        self.readout = nn.Linear(nx, 1)
        # State encoder for initial conditions of sub-sequences
        self.encoder = nn.Linear(4, nx) # takes past 2 steps of [u, y]

    def rk4_step(self, x, u_curr, u_next, dt=0.1):
        u_mid = 0.5 * (u_curr + u_next)
        k1 = self.vf(x, u_curr)
        k2 = self.vf(x + 0.5 * dt * k1, u_mid)
        k3 = self.vf(x + 0.5 * dt * k2, u_mid)
        k4 = self.vf(x + dt * k3, u_next)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forward_trajectory(self, x0, u_seq, dt=0.1):
        # u_seq: (batch, T, 1)
        batch_size, T, _ = u_seq.shape
        states = []
        x = x0
        states.append(x)
        for t in range(T - 1):
            u_curr = u_seq[:, t, :]
            u_next = u_seq[:, t + 1, :]
            x = self.rk4_step(x, u_curr, u_next, dt=dt)
            states.append(x)
        states = torch.stack(states, dim=1) # (batch, T, nx)
        y_hat = self.readout(states) # (batch, T, 1)
        return y_hat, states

def train_and_eval():
    torch.manual_seed(42)
    np.random.seed(42)
    
    u, y = load_data()
    N = len(u)
    
    # 60% Train, 40% Test split
    N_train = int(0.6 * N)
    u_train, y_train = u[:N_train], y[:N_train]
    u_test, y_test = u[N_train:], y[N_train:]
    
    u_mean, u_std = np.mean(u_train), np.std(u_train)
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    
    u_train_n = (u_train - u_mean) / (u_std + 1e-8)
    y_train_n = (y_train - y_mean) / (y_std + 1e-8)
    u_test_n = (u_test - u_mean) / (u_std + 1e-8)
    y_test_n = (y_test - y_mean) / (y_std + 1e-8)
    
    nx = 2
    hidden_dim = 32
    dt = 0.1
    model = NeuralODE(nx=nx, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-4)
    
    seq_len = 80
    batch_size = 64
    epochs = 25
    steps_per_epoch = 30
    
    print("Training baseline Neural ODE...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            # Sample random subsequences (seq_len + 2 for initial encoder context)
            idxs = np.random.randint(2, N_train - seq_len, size=batch_size)
            
            # Context for x0: past 2 steps of (u, y)
            ctx = []
            u_batches = []
            y_batches = []
            for idx in idxs:
                c = np.column_stack([u_train_n[idx-2:idx], y_train_n[idx-2:idx]]).flatten()
                ctx.append(c)
                u_batches.append(u_train_n[idx:idx+seq_len])
                y_batches.append(y_train_n[idx:idx+seq_len])
                
            ctx = torch.tensor(np.array(ctx), dtype=torch.float32)
            u_seq = torch.tensor(np.array(u_batches), dtype=torch.float32).unsqueeze(-1)
            y_seq = torch.tensor(np.array(y_batches), dtype=torch.float32).unsqueeze(-1)
            
            x0 = model.encoder(ctx)
            y_pred, _ = model.forward_trajectory(x0, u_seq, dt=dt)
            
            loss = nn.MSELoss()(y_pred, y_seq)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss/steps_per_epoch:.6f}")
            
    # Evaluation: Autonomous Free Run Rollout
    model.eval()
    with torch.no_grad():
        # Train free-run rollout
        u_tr_t = torch.tensor(u_train_n, dtype=torch.float32).view(1, -1, 1)
        # Use initial context from first 2 steps
        ctx_tr = torch.tensor(np.column_stack([u_train_n[:2], y_train_n[:2]]).flatten(), dtype=torch.float32).unsqueeze(0)
        x0_tr = model.encoder(ctx_tr)
        y_pred_tr_n, states_tr = model.forward_trajectory(x0_tr, u_tr_t, dt=dt)
        y_pred_tr = y_pred_tr_n.squeeze().numpy() * y_std + y_mean
        train_rmse = np.sqrt(np.mean((y_pred_tr - y_train) ** 2))
        
        # Test free-run rollout: start from the last state of train free run!
        x0_te = states_tr[:, -1, :]
        u_te_t = torch.tensor(u_test_n, dtype=torch.float32).view(1, -1, 1)
        y_pred_te_n, _ = model.forward_trajectory(x0_te, u_te_t, dt=dt)
        y_pred_te = y_pred_te_n.squeeze().numpy() * y_std + y_mean
        test_rmse = np.sqrt(np.mean((y_pred_te - y_test) ** 2))

    print(f"\nBaseline Results:")
    print(f"TRAIN_RMSE={train_rmse:.6f}")
    print(f"TEST_RMSE={test_rmse:.6f}")

if __name__ == '__main__':
    train_and_eval()
