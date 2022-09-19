# bar.jl
module Bar
# inputs: x, y, output: z, output should use .= to inplace assign
function add(x, y, z)
    z .= x .+ y
end

function square(x, out)
    for i in eachindex(x)
        out[i] = x[i] * x[i]
    end
    out
end

function square_grad(x, dout, dx)
    for i in eachindex(x)
        dx[i] = 2 * x[i]
    end
    for i in eachindex(x)
        dx[i] = dx[i] * dout[i]
    end
    dx
end

end