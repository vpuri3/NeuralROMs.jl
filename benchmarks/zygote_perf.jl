using Zygote, BenchmarkTools, Setfield

function zygote_perf(N)
    p = rand(N * N)
    st = (;a = rand(N), b = rand(N))

    # Lux interface
    function loss(p, st)
        P = reshape(p, (N, N))
        l = st.a' * P * st.b

        # modify state
        st = (;a = st.a .+ 1, b = st.b .+ 1)

        l, st
    end

    println()
    println("### Zygote.PULLBACK + ANON ###")
    println()

    (l, st), pb = Zygote.pullback(p -> loss(p, st), p)
    pb((one.(l), nothing))[1]

    println("# ROUND 3")
    # @btime (l, st), pb = Zygote.pullback(p -> loss(p, $st), $p)
    @btime $pb((one.($l), nothing))[1]

    println("### PULLBACK + Fix2 ###")
    loss2 = Base.Fix2(loss, st) # can be updated with @set! loss2.x = st

    @btime @set! $(loss2.x) = $st
    @btime a, pb = Zygote.pullback($loss2, $p)
    @btime gr = $pb((one.($l), nothing))[1]

    return
end

# zygote_perf(10)

###
# Tain loop
###
function loop_fix2(N, E)
    p = rand(N * N)
    st = (;a = rand(N), b = rand(N))

    # Lux interface
    function loss(p, st)
        P = reshape(p, (N, N))
        l = st.a' * P * st.b

        # modify state
        st = (;a = st.a .+ 1, b = st.b .+ 1)

        l, st
    end

    function grad(p, st)
        loss2 = Base.Fix2(loss, st)

        (l, st), pb = Zygote.pullback(loss2, p)
        gr = pb((one.(l), nothing))[1]

        l, gr, st
    end

    for _ in 1:E
        l, gr, st = grad(p, st)
        p -= 0.01 * gr
    end

    return
end

function loop_anon(N, E)
    p = rand(N * N)
    st = (;a = rand(N), b = rand(N))

    # Lux interface
    function loss(p, st)
        P = reshape(p, (N, N))
        l = st.a' * P * st.b

        # modify state
        st = (;a = st.a .+ 1, b = st.b .+ 1)

        l, st
    end

    function grad(p, st)
        (l, st), pb = Zygote.pullback(p -> loss(p, st), p)
        gr = pb((one.(l), nothing))[1]

        l, gr, st
    end

    for _ in 1:E
        l, gr, st = grad(p, st)
        p -= 0.01 * gr
    end

    return
end

const N, E = 10, 20

@btime loop_anon(N, E)
@btime loop_fix2(N, E)
