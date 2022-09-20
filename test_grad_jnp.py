import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from basic import jl_square, jl_square_grad

# 在动态图模式下开启同步执行，否则会出现Segmentation fault
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", pynative_synchronize=True)

def square(x):
    out = jl_square(x)
    return out

def square_grad(x, dout):
    dx = jl_square_grad(x, dout)
    return dx


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
        out = self.op(x)
        return out

if __name__ == "__main__":
    x = ms.Tensor(np.array([1.0, 4.0, 9.0]).astype(np.float32))
    sens = ms.Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    # net = Net()
    # calling ops is fine
    # print(bprop()(x, x, sens)[0])
    # print(net.op(x), "\n")
    dx = ops.GradOperation(sens_param=True)(Net())(x, sens)
    print(dx)