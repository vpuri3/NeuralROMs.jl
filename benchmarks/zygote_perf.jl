using Zygote, BenchmarkTools, Setfield

function zygote_anon_perf(N)
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

    println("### PULLBACK + Fix2 ###")
    loss2 = Base.Fix2(loss, st) # can be updated with @set! loss2.x = st

    println("# ROUND 1")
    @time @set! loss2.x = st
    @time (l, st), pb = Zygote.pullback(loss2, p)
    @time gr = pb((one.(l), nothing))[1]

    println("# ROUND 2")
    @time @set! loss2.x = st
    @time (l, st), pb = Zygote.pullback(loss2, p)
    @time gr = pb((one.(l), nothing))[1]

    println("# ROUND 3")
    @time @set! loss2.x = st
    @time (l, st), pb = Zygote.pullback(loss2, p)
    @time gr = pb((one.(l), nothing))[1]

    println("# BTIME")
    @btime @set! $(loss2.x) = $st
    @btime a, pb = Zygote.pullback($loss2, $p)
    @btime gr = $pb((one.($l), nothing))[1]

    println()
    println("### Zygote.PULLBACK + ANON ###")
    println()

    println("# ROUND 1")
    @time (l, st), pb = Zygote.pullback(p -> loss(p, st), p)
    @time pb((one.(l), nothing))[1]

    println("# ROUND 2")
    @time (l, st), pb = Zygote.pullback(p -> loss(p, st), p)
    @time pb((one.(l), nothing))[1]

    println("# ROUND 3")
    @time (l, st), pb = Zygote.pullback(p -> loss(p, st), p)
    @time pb((one.(l), nothing))[1]

    return
end

# zygote_anon_perf(10)


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

@btime loop_anon(10, 20)
@btime loop_fix2(10, 20)
