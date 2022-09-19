from jnumpy import init_jl, init_project

init_jl(True)
init_project(__file__)

from _basic import jl_mat_add, jl_square, jl_square_grad
