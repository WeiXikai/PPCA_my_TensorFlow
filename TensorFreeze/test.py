import numpy as np
import ctypes

test_lib = ctypes.cdll.LoadLibrary("./libtest_mkl.so")

test_lib.test()
