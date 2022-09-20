module basic

using TyPython
using TyPython.CPython
# using LoopVectorization

# @export_py function gemmavx(x::AbstractArray, y::AbstractArray)::AbstractArray
#     z = zeros(eltype(x), size(x, 1), size(y, 2))
#     @turbo for m ∈ axes(x, 1), n ∈ axes(y, 2)
#         zmn = zero(eltype(z))
#         for k ∈ axes(x, 2)
#             zmn += x[m, k] * y[k, n]
#         end
#         z[m, n] = zmn
#     end
#     return z
# end

@export_py function mat_add(a::Matrix{Float32}, b::Matrix{Float32})::Matrix{Float32}
    return a .+ b
end

@export_py function square(x::Vector{Float32})::Vector{Float32}
    out = similar(x)
    for i in eachindex(x)
        out[i] = x[i] * x[i]
    end
    out
end

@export_py function square_grad(x::Vector{Float32}, dout::Vector{Float32})::Vector{Float32}
    dx = similar(x)
    for i in eachindex(x)
        dx[i] = 2 * x[i]
    end
    for i in eachindex(x)
        dx[i] = dx[i] * dout[i]
    end
    dx
end

@export_py function square_grad!(dx::Vector{Float32}, x::Vector{Float32}, dout::Vector{Float32})::Vector{Float32}
    for i in eachindex(x)
        dx[i] = 2 * x[i]
    end
    for i in eachindex(x)
        dx[i] = dx[i] * dout[i]
    end
    dx
end

function init()
    @export_pymodule _basic begin
        jl_mat_add = Pyfunc(mat_add)
        jl_square = Pyfunc(square)
        jl_square_grad = Pyfunc(square_grad)
        jl_square_grad_in = Pyfunc(square_grad!)
    end
end

precompile(init, ())

end
