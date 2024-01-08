import mlx.core as mx
import numpy as np
import time

# calculate time
a = mx.random.normal([10000, 10000])
t1 = time.time()
a2 = mx.mean(a)
t2 = time.time()
print(a2, t2 - t1 * 10**-3)


b = np.random.normal(size=[10000, 10000])
t3 = time.time()
b2 = np.mean(b)
t4 = time.time()
print(b2, t3 - t4)
