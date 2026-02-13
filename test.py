import os
import sys

print("Python Executable:", sys.executable)
print("System Path:")
for p in sys.path:
    print(" -", p)

print("\nDang thu import tensorflow...")
try:
    import tensorflow as tf
    print("Thanh cong! Version:", tf.__version__)
except ImportError as e:
    print("\nLOI CHI TIET:")
    print(e)
except Exception as e:
    print("\nLOI KHAC:")
    print(e)