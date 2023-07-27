using Zygote, BenchmarkTools, Setfield
using MLUtils

function loop_anon(E, p, st, loader)

    function loss(x, y, p, st)
        l = sum(p * x .+ st.a, dims = 1) - y |> sum
        st = (;a = st.a .+ 1f0, b = st.b .+ 1f0)

        l, st # Lux interface
    end

    function grad(x, y, p, st)
        (l, st), pb = Zygote.pullback(p -> loss(x, y, p, st), p)
        gr = pb((one.(l), nothing))[1]

        l, gr, st
    end

    for _ in 1:E
        for (x, y) in loader
            _, gr, st = grad(x, y, p, st)
            p -= 1f-2 * gr
        end
    end

    return
end

module Losses

struct Loss{Tdata, Tst}
    data::Tdata
    st::Tst
end

function (L::Loss)(p)
    (x, y), st = L.data, L.st

    l = sum(p * x .+ st.a, dims = 1) - y |> sum
    st = (;a = st.a .+ 1f0, b = st.b .+ 1f0)

    l, st # Lux interface
end

end

import .Losses

function loop_struct(E, p, st, loader)


    function grad(loss, p)
        (l, st), pb = Zygote.pullback(loss, p)
        gr = pb((one.(l), nothing))[1]

        l, gr, st
    end

    for _ in 1:E
        for batch in loader
            loss = Losses.Loss(batch, st) # move this out

            _, gr, st = grad(loss, p)
            p -= 1f-2 * gr
        end
    end

    return
end

const N, K, E = 10, 100, 20
const p = rand(Float32, N, N)
const st = (;a = rand(Float32, N), b = rand(Float32, N))
const data = (rand(Float32, N, K), rand(Float32, 1, K))
const loader = DataLoader(data; batchsize = 10, shuffle = true)

@btime loop_anon($E, $p, $st, $loader)   # 2.508 ms (22761 allocations: 1.74 MiB)
@btime loop_struct($E, $p, $st, $loader) # 670.061 μs (7162 allocations: 1.07 MiB)
