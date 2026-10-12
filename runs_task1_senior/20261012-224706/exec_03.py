import os
import sys
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

# Let's inspect the data first and design the enhanced model
data_path = os.environ.get('SNLS_DATA', 'SNLS80mV.mat')
if not os.path.exists(data_path):
    for p in ['SNLS80mV.mat', '../SNLS80mV.mat', '/Users/mari/Desktop/Task5-6/SNLS80mV.mat']:
        if os.path.exists(p):
            data_path = p
            break

mat = scipy.io.loadmat(data_path)
u = mat['V1'].squeeze().astype(np.float32)
y = mat['V2'].squeeze().astype(np.float32)
N = len(u)
n_train = int(N * 0.6)
u_train, y_train = u[:n_train], y[:n_train]
u_test, y_test = u[n_train:], y[n_train:]

print(f"Total samples: {N}, Train: {n_train}, Test: {N - n_train}")
print(f"u_train mean: {u_train.mean():.6f}, std: {u_train.std():.6f}, min: {u_train.min():.6f}, max: {u_train.max():.6f}")
print(f"y_train mean: {y_train.mean():.6f}, std: {y_train.std():.6f}, min: {y_train.min():.6f}, max: {y_train.max():.6f}")

# Check exec_01.py to see how it was written
with open('exec_01.py', 'r') as f:
    exec_01_code = f.read()

print("--- exec_01.py snippet ---")
print(exec_01_code[:1200])
