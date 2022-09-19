import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops

ms.set_context(device_target="CPU")


# 反向传播函数
def bprop():
    op = ops.Custom("./bar.jl:Bar:square_grad", lambda x, _: x, lambda x, _: x, func_type="julia")

    def custom_bprop(x, out, dout):
        dx = op(x, dout)
        return (dx,)

    return custom_bprop

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.op = ops.Custom("./bar.jl:Bar:square", lambda x: x, lambda x: x, bprop=bprop(), func_type="julia")

    def construct(self, x):
        return self.op(x)

if __name__ == "__main__":
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    dx = ops.GradOperation(sens_param=True)(Net())(ms.Tensor(x), ms.Tensor(sens))
    print(dx)