import numpy as np

a = (1, 2, 3, 4)

print ((np.random.rand(*a) < 0.5) / 0.5)
