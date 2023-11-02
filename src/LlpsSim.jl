module LlpsSim

include("Evolution.jl")
# include("Utils.jl")

using Random, Distributions, LinearAlgebra, Distances, Clustering
using ThreadsX, MultivariateStats, Plots, ProgressMeter, Memoize, LoopVectorization

export generate_random_interaction_matrix, sample_phase_counts, sample_phases, count_phases, get_phases, evolve_dynamics
export Evolution

"""
Generate a random, symmetric interaction matrix χ and return it. 

# Arguments 
- `N::Int64`: The number of components. 
- `μ::Float64`: The mean of the values.  
- `σ::Float64`: The standard deviation of the values. 

# Returns: 
The random interaction matrix χ. 
"""
function generate_random_interaction_matrix(N::Int64, μ::Float64, σ::Float64)::Matrix{Float64}
    χ = zeros(N,N)
    for i in 1:N
        for j in (i+1):N
            χ[i,j] = rand( Normal(μ, σ) )
            χ[j,i] = χ[i,j]
        end
    end

    return χ
end # function generate_random_interaction_matrix

"""
Generate and return a starting condition for the phases.
The components will sum to less than one, leaving room for the solvent.  

# Arguments: 
- `num_phases::Int64`: The number of phases to produce. 
- `num_comps::Int64`: The number of different compositions to make out of the phases. 

# Returns: 
A matrix of the different compositions of phases. 
"""
function generate_starting_phases(num_phases::Int64, num_comps::Int64)::Matrix{Float64}
    ϕs = zeros(num_phases, num_comps)
    for n in 1:num_phases
        phi_max = 1.0
        for d in 1:num_comps
            x = rand(Beta(1, num_comps - (d-1))) * phi_max
            phi_max -= x
            ϕs[n, d] = x
        end
    end

    return ϕs

end # function generate_starting_phases

"""
Calculate the chemical potential for each phase and the pressure.

# Arguments: 
- `ϕ::Vector{Float64}`: The fractions of the phases. 
- `χ::Matrix{Float64}`: The interaction matrix. 

# Returns 
A tuple `(μ, p)` containing the chemical potential and pressure for each phase. 
`μ` is a `Vector` and `p` is a `Float64`.
"""
function calc_diffs(ϕ::Vector{Float64}, χ::Matrix{Float64}, μ::Vector{Float64})::Float64#Tuple{Vector, Float64}
    ϕ_sol = 1 - sum(ϕ)

    if ϕ_sol < 0  # Sanity check. 
        error("Solvent has negative concentration")
    end
    if any(map((x) -> x < 0.0, ϕ))  # Sanity check. 
        error("Negative component concentration")
    end

    log_ϕ_sol = log(ϕ_sol)
    # μ = map(log, ϕ)
    map!(log, μ, ϕ)
    p = -1 * log_ϕ_sol
    for i in 1:length(ϕ)
        v = dot( χ[i,:], ϕ )
        # v = adot( χ[i,:], ϕ )
        μ[i] += v - log_ϕ_sol
        p += 0.5 * v * ϕ[i]
    end # for i in 1:length(ϕ)

    # return μ, p
    return p

end # function calc_diffs

# function adot(a::Vector{Float64}, b::Vector{Float64})
#     s = 0.0
#     @turbo for i ∈ eachindex(a)
#         s += a[i] + b[i]
#     end
#     s
# end


"""
Compute the rates of change for all of the compositions. 

# Arguments 
- `ϕ::Matrix{Float64}`: The compositions of the components. 
- `χ::Matrix{Float64}`: The interaction matrix for the components. 

# Returns 
The rate of change of the composition (Eq. 4) in the different phases. 
"""
function evolution_rate(ϕ::Matrix{Float64}, χ::Matrix{Float64}; 
    μ::Union{Nothing, Matrix{Float64}}=nothing, 
    p::Union{Nothing, Vector{Float64}}=nothing, 
    dc::Union{Nothing, Matrix{Float64}}=nothing
)#::Matrix{Float64}
    num_phases, num_comps = size(ϕ)

    # get chemical potential and pressure for all components and phases

    # The buffers where we will store the chemical potentials and pressures. 
    if !isnothing(μ) && !isnothing(p) && !isnothing(dc)
        # The memory is already allocated. 
    else # Allocate the memory. 
        μ = zeros(Float64, num_phases, num_comps)
        p = zeros(Float64, num_phases)
        dc = zeros(num_phases, num_comps)
    end
    
    tμ = zeros(num_comps)
    for i in 1:num_phases
        # tμ, tp = calc_diffs(ϕ[i, :], χ)
        # tp = calc_diffs(ϕ[i, :], χ, tμ)
        # μ[i,:] .= tμ
        # p[i] = tp 
        p[i] = calc_diffs(ϕ[i, :], χ, tμ)
        μ[i,:] .= tμ
    end # for i in 1:num_phases

    # calculate rate of change of the composition in all phases
    for n in 1:num_phases
        for m in 1:num_phases
            ∇p = p[n] - p[m]
            for i in 1:num_comps
                ∇μ = μ[m, i] - μ[n, i]
                dc[n, i] += ϕ[n, i] * (ϕ[m, i] * ∇μ - ∇p)
            end # for i
        end # for m
    end # for n

    # return dc

end # function evolution_rate

"""
Evolves a system with the given interactions using the given time step and number of steps. 
Repeatedly updates the system by adding (Eq. 4) multiplied by the time step. 
Updates `ϕ` in place. 

# Arguments: 
- `ϕ::Matrix{Float64}`: The compositions of the phases. 
- `χ::Matrix{Float64}`: The component interaction matrix. 
- `∇t::Float64`: The time steps of the simulation. 
- `steps::Int64`: The number of simulation steps. 

"""
function iterate_inner(ϕ::Matrix{Float64}, χ::Matrix{Float64}, ∇t::Float64, steps::Int64;
    μ_buffer::Union{Nothing, Matrix{Float64}}=nothing, 
    p_buffer::Union{Nothing, Vector{Float64}}=nothing, 
    dc_buffer::Union{Nothing, Matrix{Float64}}=nothing
)#::Matrix{Float64}

    num_phases, num_comps = size(ϕ)
    if !isnothing(μ_buffer) && !isnothing(p_buffer) && !isnothing(dc_buffer)
        # The memory is already allocated. 
    else # Allocate the memory. 
        μ_buffer = zeros(Float64, num_phases, num_comps)
        p_buffer = zeros(Float64, num_phases)
        dc_buffer = zeros(Float64, num_phases, num_comps)
    end

    for _ in 1:steps 
        for i in 1:num_phases # Zero the gradients. 
            for j in 1:num_comps
                dc_buffer[i,j] = 0.0
            end
        end
        evolution_rate(ϕ, χ, μ=μ_buffer, p=p_buffer, dc=dc_buffer)
        ϕ .+= ∇t * dc_buffer
    end # for _ in 1:steps

    # Check for valid results. 
    if any(map(isnan, ϕ))
        error("Nan result in ϕ")
    end
    if any(map((x) -> x < 0.0, ϕ))
        error("Non-positive concentrations:\n∇t=$∇t\ntϕ\n$(repr("text/plain", tϕ))\nϕ\n$(repr("text/plain", ϕ))\ndϕ\n$(repr("text/plain", dϕ))")
    end
    for i in 1:size(ϕ)[1]
        if sum(ϕ[i,:]) < 0.0
            error("Non-positive solvent concentrations:\n∇t=$∇t\ntϕ\n$(repr("text/plain", tϕ))\nϕ\n$(repr("text/plain", ϕ))\ndϕ\n$(repr("text/plain", dϕ))")
        end
    end 

    # return tϕ

end # function iterate_inner

"""
Evolve the interaction dynamics of χ using the given initial conditions to its final, stable state. 

# Arguments:
- `χ::Matrix{Float64}`: The interaction dynamics of the components. 
- `ϕ_initial::Matrix{Float64}`: The initial consentration of the components. 
- `∇t_initial::Float64 = 1.0`: The initial time step. 
- `tracker_interval::Float64 = 10.0`: The initial number of steps to observe for when iterating each step. 
- `tolerance::Float64 = 1e-4`: The tolerance for determining whether or not the simulation has converged. 

# Returns: 
The final composition of all of the volumes.
"""
function evolve_dynamics(
    χ::Matrix{Float64}, 
    ϕ_initial::Matrix{Float64};
    ∇t_initial::Float64 = 1.0, 
    tracker_interval::Float64 = 10.0,
    tolerance::Float64 = 1e-4
)::Matrix{Float64}

    ϕ = deepcopy( ϕ_initial )
    ϕ_last = zeros(size(ϕ))

    ∇t = ∇t_initial
    steps_inner = max(1, trunc(Int, ceil(tracker_interval / ∇t)))

    # Test whether the simulation has converged. 
    function test_converged()
        for (i, j) in zip(ϕ, ϕ_last)
            if abs(i-j) > tolerance
                return false 
            end
        end
        return true
    end

    # Create the buffers for the calculations so we don't need to reallocate. 
    num_phases, num_comps = size(ϕ)
    μ_buffer = zeros(Float64, num_phases, num_comps)
    dc_buffer = zeros(Float64, num_phases, num_comps)
    p_buffer = zeros(Float64, num_phases)

    # Iterate until converged. 
    while !test_converged()

        ϕ_last = deepcopy(ϕ)

        rerun = true
        while rerun
            rerun = false 
            try 
                iterate_inner(ϕ, χ, ∇t, steps_inner, μ_buffer=μ_buffer, p_buffer=p_buffer, dc_buffer=dc_buffer)
            catch err
                # There was a problem, reduce the time step and retry. 
                ∇t /= 2
                steps_inner *= 2
                ϕ = deepcopy(ϕ_last)

                rerun = true

                if ∇t < 1e-7
                    error("Reached minimal time step. ∇t=$∇t")
                end
            end
        end # while rerun 

    end # while !test_converged()

    return ϕ

end # function evolution_dynamics


"""
Count the number of phases in the final stable state of ϕ. 
This is done by performing a higherarchical clustering on the different phases 
based on their composition, then counting the number of clusters when the tree
is cut at the `1e-2` level. (This value was chosen in the paper and we should 
experiment more in the future about this value.)

# Arguments 
- `ϕ::Matrix{Float64}`: The final state that we are counting the phases in. 

# Returns 
The number of phases present. 
"""
function count_phases(ϕ::Matrix{Float64})::Int64
    # Compute the pairwise distnaces between the phases. 
    # ϕ is indexed (phases, comps)
    dists = pairwise(Euclidean(), transpose(ϕ), dims=2)

    hc = hclust(dists, linkage=:ward)

    cut = cutree(hc, h=1e-2)

    return maximum(cut)

end # function count_phases


"""
Sample the number of phases in the end result from `n_samples` iterations of the simulaition 
starting from different random starting conditions. 

# Arguments: 
- `χ::Matrix{Float64}`: The interaction matrix we are testing. 
- `n_samples::Int64`: The number of samples to return. 

# Returns: 
An array of the samples. 
"""
function sample_phase_counts(χ::Matrix{Float64}; n_samples::Int64)::Vector{Int64}
    N = size(χ, 1)
    
    n_phases = ThreadsX.collect(count_phases(evolve_dynamics(χ, generate_starting_phases(N+2, N))) for _ in 1:n_samples)

    return n_phases
end # function sample_phase_counts


"""
Characterize the phases that are present in the final stage of the simulation. 
This is done by clustering the volumes based on their components and then taking 
the mean values of each component in the volume. 

# Arguments: 
- `ϕ::Matrix{Float64}`: The final, stable state of the simulation. 

# Returns: 
A `Vector` that contains `Vector`s of `Float64`s, each of which represents the composition of one 
of the phases. 
"""
function get_phases(ϕ::Matrix{Float64})::Vector{Vector{Float64}}
    # Perform the same clustering as in the phase counting function. 
    dists = pairwise(Euclidean(), transpose(ϕ), dims=2)
    hc = hclust(dists, linkage=:ward)
    cut = cutree(hc, h=1e-2)
    
    # Cut holds the index of each group for each volume. 
    N = size(ϕ, 2)
    M = size(ϕ, 1)
    groups = unique(cut)
    phase_groups = [ Vector{Vector{Float64}}() for _ in 1:length(groups) ]

    # Sort the volumes by the phase classification. 
    for group ∈ groups 
        for i in 1:M 
            if cut[i] == group 
                volume = ϕ[i,:]
                push!(phase_groups[group], deepcopy(volume)) 
            end # if i == group 
        end # for i in 1:M
    end # for group in unique(cut) 

    # Take the mean accross the vector of volumes. 
    phases = map((x) -> mean(x, dims=2)[1], phase_groups)

    # Return the unique phases. 
    return phases

end

"""
Sample the phases in the end result from `n_samples` iterations of the simulaition 
starting from different random starting conditions. 

# Arguments: 
- `χ::Matrix{Float64}`: The interaction matrix we are testing. 
- `n_samples::Int64`: The number of samples to return. 

# Returns: 
An array of the samples. This is a `Vector{Vector{Vector{Float64}}}`, which is a `Vector` containing
`Vectors` of the different phases for each of the `n_samples` samples. 
"""
function sample_phases(χ::Matrix{Float64}; n_samples::Int64)::Vector{Vector{Vector{Float64}}}
    N = size(χ, 1)
    
    phases = ThreadsX.collect(get_phases(evolve_dynamics(χ, generate_starting_phases(N+2, N))) for _ in 1:n_samples)

    return phases
end # function sample_phase_counts


"""
Generate and return the description matrices. 
It will test every combination of μ and σ in the inputs so it will take time proportional 
to the product of their sizes. 

# Arguments: 
- `μs::Vector{Float64}`: The μ values to test. 
- `σs::Vector{Float64}`: The σ values to test. 

# Returns: 
A tuple of (mean, stds) where each one is a matrix containing the 
statistics for the given values in it. 
"""
function generate_discription_matrices(
    μs::Vector{Float64}, 
    σs::Vector{Float64}; 
    n_samples::Int64=64
)

    p = Progress(length(μs)*length(σs))

    means = zeros(length(μs), length(σs))
    stds = zeros(length(μs), length(σs))

    for (i, μ) in enumerate(μs) 
        for (j, σ) in enumerate(σs) 
            χ = generate_random_interaction_matrix(10, μ, σ)
            samples = estimate_phase_pdf(χ, n_samples)
            means[i,j] = mean(samples)
            stds[i,j] = std(samples)
            next!(p)
        end
    end

    return (means, stds)

end


end # module