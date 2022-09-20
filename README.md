mindspore的文档里有注明，基于自定义的Custom算子，[julia类型](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/operation/op_custom.html#julia%E7%B1%BB%E5%9E%8B%E7%9A%84%E8%87%AA%E5%AE%9A%E4%B9%89%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91)（mindspore内置的模式）仅支持linux平台（不确定）

jnumpy在对接Custom的接口是，采用的是[pyfunc类型](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/operation/op_custom.html#julia%E7%B1%BB%E5%9E%8B%E7%9A%84%E8%87%AA%E5%AE%9A%E4%B9%89%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91)的自定义算子。但是在定义具有反向传播函数的算子时，计算梯度会出现Segmentation fault(在动态图和同步执行的模式下可以运行，其他模式的问题待解决)。

在`bench_jnumpy.py`和`bench_msjulia.py`文件里分别使用jnumpy和mindspore提供的julia类型自定义算子（放在一起定义会Segmentation fault), 在一个简单的算子(square)上测试latency和运行时的效率，jnumpy在第一次运行时需要较高的开销(除此之外还有较慢的初始化)，但在运行时效率较高。

```
python bench_jnumpy.py
# time cost of first run: 0.5339596271514893
# time cost of 1000 times run: 0.17962861061096191

python bench_msjulia.py
# time cost of first run: 0.13610386848449707
# time cost of 1000 times run: 1.2894513607025146
```