import os
import scipy.io

data_path = os.environ.get('SNLS_DATA')
print("Data path:", data_path)
mat = scipy.io.loadmat(data_path)
print("Keys:", mat.keys())
print("V1 shape:", mat['V1'].shape)
print("V2 shape:", mat['V2'].shape)
