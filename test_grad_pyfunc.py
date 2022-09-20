import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops


ms.set_context(device_target="CPU")

def square(x):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i] * x[i]
    return out
    # return x**2

def square_grad(x, dout):
    dx = np.zeros_like(x)
    for i in range(x.shape[0]):
        dx[i] = 2 * x[i]
    for i in range(x.shape[0]):
        dx[i] = dx[i] * dout[i]
    return dx
    # return 2*x*dout


# 反向传播函数
def bprop():
    op = ops.Custom(square_grad, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="pyfunc")

    def custom_bprop(x, out, dout):
        dx = op(x, dout)
        return (dx,)

    return custom_bprop

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.op = ops.Custom(square, lambda x: x, lambda x: x, bprop=bprop(), func_type="pyfunc")

    def construct(self, x):
        return self.op(x)

if __name__ == "__main__":
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    dx = ops.GradOperation(sens_param=True)(Net())(ms.Tensor(x), ms.Tensor(sens))
    print(dx)