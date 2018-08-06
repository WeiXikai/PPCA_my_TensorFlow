# import ctypes
# import numpy as np
# lib = ctypes.cdll.LoadLibrary("./test.so")
#
# test = lib.test_array
#
# a = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]]], dtype=np.float32)
#
# b = np.array(np.rot90(a, 2, (1, 2)))
#
# test(b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 4, 2, 2)

cnt = 0

for i in range(10):
    for j in range (10):
        for k in range(15):
            for x in range(100):
                for y in range(100):
                    cnt += 1
