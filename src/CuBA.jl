include("BA.jl")
import LinearAlgebra
import CUDA


function run_RD_CUDA(name::String, p, d, betas::Vector{F}, T::Int, device_id::Int) where F<:AbstractFloat
    CUDA.device!(device_id)
    #CBs = []
    diff = Float32[]
    Rs = Float32[]
    Ds = Float32[]
    p = Float32.(p)
    d = Float32.(d)
    betas = Float32.(betas)
    p = CUDA.CuArray(p)
    d = CUDA.CuArray(d)
    CB_prev, R, D = CuBA(p, d, betas[1], T)
    CB_prev = Array(CB_prev)
    push!(Rs, R)
    push!(Ds, D)
    for l in ProgressBars.ProgressBar(2:length(betas))
        #println(l)
        beta = betas[l]
        CB, R, D = CuBA(p, d, beta, T)
        #println(typeof(CB), " ", typeof(R), " ", typeof(D))
        # save CB, R and D then write a python script to read/plot them
        CB = Array(CB)
        push!(diff, CB_diff_KL(CB_prev, CB))
        CB_prev = CB
        push!(Rs, R)
        push!(Ds, D)
    end
    _, _, crit_inds = get_possible_peaks(βs, diff)
    crit_inds = Int.(crit_inds)
    println(diff)
    println(crit_inds)
    CBs = []
    for i in ProgressBars.ProgressBar(crit_inds)
        CB, _, _ = CuBA(p, d, betas[i], T)
        push!(CBs, Array(CB))
    end
    CBs = Matrix{Float32}.(CBs)
    Rs = vcat(Rs...)
    Ds = vcat(Ds...)
    RDresult(name, betas, CBs, Rs, Ds, Array(p), Array(d), diff, crit_inds)
end


function CuBA(p, d, β::F, num::Int) where {F<:AbstractFloat}
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
    D = sum(D[.!isnan.(D)])
    tmp = p .* log.(Z)
    tmp = sum(tmp[.!isnan.(tmp)])
    R = (-β * D - tmp) ./ log(2)
    return P, R, D
end


if abspath(PROGRAM_FILE) == @__FILE__
    settings = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table settings begin
        "device_id"
            help = "Device id starting from 0"
            arg_type = Int
            required = true
        "scale"
            help = "Downscale to get RD curve from, starts from 0"
            arg_type = Int
            required = true
    end
    args = ArgParse.parse_args(settings)
    device_id = args["device_id"]
    scale = args["scale"]

    start = -20
    stop = log(150)
    divs = 1000
    diff = (stop - start) / (divs-1)
    βs = exp.([start + i*diff for i in 0:(divs-1)])
    T = 10000

    base_dir = "../save/Inter4K"
    save_dir = base_dir
    prepend_save_name = "UHD_2"
    num_bits = 16
    num_scales = 6
    num_pixels = floor(Int64, num_bits / 2)
    num_states = 3 ^ num_pixels
    println("Running $scale on device $device_id")
    local dist = create_ternary_dist_mat(num_bits)
    for sc in scale:scale#0:num_scales-1
        p =  NPZ.npzread("$(base_dir)/$(prepend_save_name)_$(sc)xdown_probs.npy")
        result = run_RD_CUDA("$(prepend_save_name)_$(sc)xdown_RD.npy", p, dist, βs, T, device_id)
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
    res = run_RD_CUDA("test", p, d, βs, T, 0)
    for β in βs
        q, r, ds = CuBA(p, d, β, T)
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


