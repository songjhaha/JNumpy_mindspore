import numpy as np
import mindspore as ms
import mindspore.ops as ops
from basic import jl_square
import time


ms.set_context(device_target="CPU")

def square(x):
    return jl_square(x)


# op1 = ops.Custom("./bar.jl:Bar:square", lambda x: x, lambda x: x, func_type="julia")
op2 = ops.Custom(square, lambda x: x, lambda x: x, func_type="pyfunc")

def test_time(op):
    start = time.time()
    for i in range(1000):
        x = np.random.rand(10000).astype(np.float32)
        output = op(ms.Tensor(x))
    end = time.time()
    return end - start

x = np.random.rand(10000).astype(np.float32)
start = time.time()
op2(ms.Tensor(x)) # avoid latency
print("time cost of first run:", time.time() - start)

print("time cost of 1000 times run:", test_time(op2))
