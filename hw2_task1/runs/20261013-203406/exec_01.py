import os
import numpy as np
import scipy.io as sio

# Load data
data_path = os.environ.get('SNLS_DATA')
if not data_path:
    raise EnvironmentError("SNLS_DATA environment variable not set")
mat = sio.loadmat(data_path)
# Assuming variables are named V1 and V2 as per description
u_raw = mat['V1'].flatten()
y_raw = mat['V2'].flatten()

# Excerpt indices
start = 42650
end = 46746  # exclusive in Python slicing, but we want inclusive up to 46745? Actually slice [42650 : 46746] gives 4096 samples: 46746-42650=4096.
excerpt = slice(start, end)  # Python slice end exclusive
u_excerpt = u_raw[excerpt]
y_excerpt = y_raw[excerpt]

# Center
u = u_excerpt - np.mean(u_excerpt)
y = y_excerpt - np.mean(y_excerpt)

# Sampling frequency
fs = 610.35  # Hz
dt = 1.0 / fs

# Central differences for interior points
# We'll compute y_dot and y_ddot for i=1..N-2
N = len(u)
# Ensure we have at least 3 points
if N < 3:
    raise ValueError("Excerpt too short for differentiation")
# Allocate arrays of length N-2
y_dot = np.zeros(N-2)
y_ddot = np.zeros(N-2)
# Central difference
for i in range(1, N-1):
    y_dot[i-1] = (y[i+1] - y[i-1]) / (2.0 * dt)
    y_ddot[i-1] = (y[i+1] - 2.0*y[i] + y[i-1]) / (dt**2)

# Now u, y, y_dot, y_ddot all correspond to indices 1..N-2 of original excerpt
# Align them: take u[1:-1], y[1:-1]
u_aligned = u[1:-1]
y_aligned = y[1:-1]

# Build design matrix X: columns for -y_dot, -y, -(y^3), u
X = np.column_stack((-y_dot, -y_aligned, -(y_aligned**3), u_aligned))
# Target vector z = y_ddot
z = y_ddot

# Solve least squares
theta, residuals, rank, s = np.linalg.lstsq(X, z, rcond=None)
a_hat, b_hat, c_hat, g_hat = theta

# Compute ratio |c|*y^2/|b| at y=0.2 V
y_peak = 0.2
ratio = np.abs(c_hat) * (y_peak**2) / np.abs(b_hat)

# Print results
print(f"a_hat = {a_hat:.6f} s^-1")
print(f"b_hat = {b_hat:.6f} s^-2")
print(f"c_hat = {c_hat:.6f} V^-2 s^-2")
print(f"g_hat = {g_hat:.6f}")
print(f"|c|y^2/|b| at |y|=0.2 V = {ratio:.6f}")