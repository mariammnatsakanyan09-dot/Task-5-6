import os
import sys

# Check current files
print("Current directory:", os.getcwd())
if os.path.exists("exec_01.py"):
    with open("exec_01.py", "r") as f:
        print("exec_01.py contents:")
        print(f.read()[:1500])
else:
    print("exec_01.py does not exist yet in cwd.")

data_path = os.environ.get('SNLS_DATA', '/Users/mari/Desktop/Task5-6/SNLS80mV.mat')
print("Data path:", data_path, "exists:", os.path.exists(data_path))
