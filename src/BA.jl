struct RDresult{F<:AbstractFloat}
    name::String
    βs::Vector{F}
    CB::Vector{Matrix{F}}
    R::Vector{F}
    D::Vector{F}
    p::Vector{F}
    d::Matrix{F}
    diff::Vector{F}
    crit_inds::Vector{Int}
end


function run_RD(name::String, p::AbstractVector, d::AbstractMatrix, βs::Vector{F}, T::Int) where F<:AbstractFloat
    CBs = Vector{Matrix{F}}[Matrix{F}[] for _ in 1:Threads.nthreads()]
    Rs = Vector{F}[F[] for _ in 1:Threads.nthreads()]
    Ds = Vector{F}[F[] for _ in 1:Threads.nthreads()]
    inds = Vector{Int}[[] for _ in 1:Threads.nthreads()]
    Threads.@threads for l in ProgressBars.ProgressBar(1:length(βs))
        β = βs[l]
        CB, R, D = BA(p, d, β, T)
        # save CB, R and D then write a python script to read/plot them
        push!(CBs[Threads.threadid()], CB)
        push!(Rs[Threads.threadid()], R)
        push!(Ds[Threads.threadid()], D)
        push!(inds[Threads.threadid()], l)
    end
    inds = collect(Iterators.flatten(inds))
    CBs = collect(zip(inds, Iterators.flatten(CBs)))
    CBs = last.(sort(CBs, by=x->x[1]))
    diff = [CB_diff_KL(CBs[i], CBs[i+1]) for i in 1:length(CBs)-1]
    _, _, crit_inds = get_possible_peaks(βs, diff)
    crit_inds = Int.(crit_inds)
    CBs = CBs[crit_inds]
    Rs = last(zip(sort(zip(inds, vcat(Rs...)) |> collect, by=x->x[1])...))
    Rs = [i for i in Rs]
    Ds = last(zip(sort(zip(inds, vcat(Ds...)) |> collect, by=x->x[1])...))
    Ds = [i for i in Ds]
    RDresult(name, βs, CBs, Rs, Ds, p, d, diff, crit_inds)
end


function save(resdir::String, res::RDresult)
    resdict = Dict(
            :betas=>res.βs,
            :CB=>vcat(res.CB...),
            :R=>res.R,
            :D=>res.D,
            :p=>res.p,
            :d=>res.d,
            :diff=>res.diff,
            :crit_inds=>res.crit_inds
    )
    NPZ.npzwrite(joinpath(resdir, res.name); resdict...)
end


function read_RDresult(respath::String)
    resdict = NPZ.npzread(respath)
    npointbynstate, num_state = size(resdict["CB"])
    CB = [resdict["CB"][i:i+num_state-1, :] for i in 1:num_state:npointbynstate]
    RDresult(respath, resdict["betas"], CB, resdict["R"], resdict["D"], resdict["p"], resdict["d"], resdict["diff"], resdict["crit_inds"])
end


function BA(p::AbstractVector , d::AbstractMatrix, β::F, num::Int) where F<:AbstractFloat
    Q = exp.(-β*d)
    P = LinearAlgebra.I
    for i in 1:num
        q = P' * p
        Z = Q * q
        P = Q .* (1.0 ./ Z * q')
    end
    q = P' * p
    Z = Q * q
    D = (P .* d)' * p
    replace!(x -> isnan(x) ? 0 : x, D)
    D = sum(D)
    tmp = p .* log.(Z)
    replace!(x -> isnan(x) ? 0 : x, tmp)
    R = (-β * D - sum(tmp)) ./ log(2)
    return P, R, D
end


function newBA(p::AbstractVector , d::AbstractMatrix, β::F, num::Int) where F<:AbstractFloat
    Q = exp.(-β*d)
    q = rand(size(p)...)
    for i in 1:num
        Z = Q * q
        t = Q' * (p ./ Z)
        q = q .* t
    end
    Z = Q * q
    P = Q .* (1.0 ./ Z * q')
    D = (P .* d)' * p
    replace!(x -> isnan(x) ? 0 : x, D)
    D = sum(D)
    tmp = p .* log.(Z)
    replace!(x -> isnan(x) ? 0 : x, tmp)
    R = (-β * D - sum(tmp)) ./ log(2)
    P, R, D
end


function CB_diff(cb1::AbstractMatrix, cb2::AbstractMatrix)
    diff = cb1 - cb2
    return Statistics.mean(abs.(diff)) / Statistics.std(diff)
end


"""
Every row of the codebook is a conditional distribution, so we can take the KL divergence between subsequent codebooks.
Not symmetric.
"""
function CB_diff_KL(cb1::AbstractMatrix, cb2::AbstractMatrix)
    total = 0.
    for i in 1:size(cb1)[1]
        cb1v = @view cb1[i, :]
        cb2v = @view cb2[i, :]
        total += diff_KL(cb1v, cb2v)
    end
    return total
end


function diff_KL(dist1::AbstractVector, dist2::AbstractVector)
    eps = floatmin(Float32)
    adj_dist1 = dist1[:]
    adj_dist1[adj_dist1 .== 0] .= eps
    dist2[dist2 .== 0] .= eps
    sum(dist1 .* log.(adj_dist1 ./ dist2))
end


@enum STATE climb=1 climb_flat=2 flat=3 fall_flat=4 fall=5


function sorted_supremum_index(val, values; start=2)
    for i in start:length(values)-1
        if val < values[i]
            return i - 1
        end
    end
    return length(values) - 2 # idk if this is needed
end


function sorted_infimum(val, values; start=1)
    for i in length(values):-1:start
        if val >= values[i]
            return values[i+1]
        end
    end
    return values[1]
end


function nearest_val_index(val, values)
    diffs = values .- val
    last(findmin(abs.(diffs)))
end


function get_possible_peaks(x::AbstractVector, y::AbstractVector; step_size=0.005, momentum=0.3, grad_thresh=1e-3) # Do not set momentum too high or you may land out of bounds
    y_prime = (y[2:end] .- y[1:end-1]) #./ (x[2:end] .- x[1:end-1])
    y_prime[abs.(y_prime) .< grad_thresh] .= 0
    sign_y_prime = sign.(y_prime)
    diff_sign = sign_y_prime[2:end] .- sign_y_prime[1:end-1]

    # states: climbing, falling, flat climb, flat fall, flat (only possible at beginning)
    state = sign_y_prime[1] == 0 ? flat : (sign_y_prime[1] > 0 ? climb : fall)
    cnt = 0
    maybevalley_ind = []
    maybepeak_ind = []
    for i in 1:length(diff_sign)
        if diff_sign[i] == 0
            if state == fall_flat || state == climb_flat
                cnt += 1
            end
        elseif diff_sign[i] == 1
            if state == fall
                state = fall_flat
            elseif state == fall_flat
                state = climb
                push!(maybevalley_ind, i+1) # i+1 - Int(ceil(cnt / 2)) is the middle of the valley
            elseif state == climb_flat
                state = climb
                cnt = 0
            elseif state == flat
                state = climb
            end
        elseif diff_sign[i] == -1
            if state == climb
                state = climb_flat
            elseif state == climb_flat
                state = fall
                push!(maybepeak_ind, i+1) # i+1 - Int(ceil(cnt / 2)) is middle of peak
            elseif state == fall_flat
                state = fall
                cnt = 0
            elseif state == flat
                state = fall
            end
        elseif diff_sign[i] == 2
            state = climb
            push!(maybevalley_ind, i+1) # valley is at i + 1
        elseif diff_sign[i] == -2
            state = fall
            push!(maybepeak_ind, i+1) # peak is at i+1
        end
    end


    # ok I think I need some sort of modified momentum on this "grid"
    peak_ind = []
    start_i = findfirst(x->x!=0, sign_y_prime)
    i = start_i
    if y_prime[i] < 0
        i = maybevalley_ind[1]
    end
    ex = x[i]
    maybepeak_x = x[maybepeak_ind]
    while true
        change = 0
        for _ in 1:500
            ϕ = ex + momentum * change
            i = sorted_supremum_index(ϕ, x) # index of largest value in x that is smaller than ϕ

            change = momentum * change - step_size * y_prime[i]
            ex -= change
        end
        # println(x[nearest_val_index(ex, x)])
        # println(maybepeak_ind[nearest_val_index(ex, maybepeak_x)])
        pind = maybepeak_ind[nearest_val_index(ex, maybepeak_x)] # index of nearest peak in y
        push!(peak_ind, pind)
        if length(maybevalley_ind) == 0 || start_i >= maybevalley_ind[end]
            break
        else
            start_i = sorted_infimum(start_i, maybevalley_ind) # index of next valley after this peak
            i = start_i
            ex = x[i]
        end
    end
    peak_ind = unique(peak_ind)

    return x[peak_ind], y[peak_ind], peak_ind
end


function create_ndiv2_ternary_dataset(n::Int)
    num_pixels = floor(Int, n/2)
    num_states =  3 ^ num_pixels
    res = zeros(Int8, num_states, num_pixels)
    choose_from = Int8[0, 1, 2]
    for (i, comb) in enumerate(Iterators.product([choose_from for _ in 1:num_pixels]...))
        for (j, c) in enumerate(comb)
            res[i, j] = c
        end
    end
    return res, num_states
end


function create_ternary_dist_mat(num_bits::Int)
    dset_inds, num_states = create_ndiv2_ternary_dataset(num_bits)
    d = Matrix{Float64}(undef, num_states, num_states)
    Threads.@threads for i in 1:num_states
        for j in 1:num_states
            s1 = dset_inds[i, :]
            s2 = dset_inds[j, :]
            #d[i, j] = sum(count_ones(a ⊻ b) for (a, b) in zip(s1, s2))
            d[i, j] = sum(count_ones.(s1 .⊻ s2))
        end
    end
    return d
end


if abspath(PROGRAM_FILE) == @__FILE__
    start = -20
    stop = log(150)
    divs = 1000
    diff = (stop - start) / (divs-1)
    βs = exp.([start + i*diff for i in 0:(divs-1)])
    T = 10000

    #d = create_ternary_dist_mat(8)
    ##d = NPZ.npzread("../../../RD/2x2distortion.npz")["arr_0"]
    #p = NPZ.npzread("../save/vanhateren/iml_2_0xdown_probs.npy")
    #result = run_RD("moo.npz", p, d, βs, T)
    #save_RDresult(".", result)
    #exit()


    base_dir = "../save/vanhateren"
    save_dir = base_dir
    prepend_save_name = "iml_2"
    num_bits = 8

    num_pixels = floor(Int64, num_bits / 2)
    num_scales = 9
    order = num_pixels + 1
    d = create_ternary_dist_mat(num_bits)
    for scale in 0:num_scales-1
        local p =  NPZ.npzread("$(base_dir)/$(prepend_save_name)_$(scale)xdown_probs.npy")
        result = run_RD("$(prepend_save_name)_$(scale)xdown_RD.npy", p, d, βs, T)
        save_RDresult(save_dir, result)
    end

    exit()

    d = Float64.([[0, 1, 2] [1, 0, 1] [2, 1, 0]])
    d = Array(d')
    p = [1/13, (1-1/13)/2, (1-1/13)/2]
    T = 1000
    start = 0.01
    stop = 10
    diff = stop - start
    div = 100
    βs = [start+i*diff/(div-1) for i in (div-1):-1:0]
    Ps = []
    Rs = []
    Ds = []
    res = run_RD("test", p, d, βs, T)
    for β in βs
        q, r, ds = BA(p, d, β, T)
        append!(Ps, q)
        append!(Rs, r)
        append!(Ds, ds)
    end
    println(Rs)
    println(Ds)
    println(βs)
    println(Rs == res.R)
    println(Ds == res.D)
end

