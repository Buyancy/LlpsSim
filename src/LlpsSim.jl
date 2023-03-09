module LlpsSim

include("Evolution.jl")

using Random, Distributions, LinearAlgebra, Distances, Clustering
using ThreadsX, MultivariateStats, Plots, ProgressMeter, Memoize

export compute_phases, generate_random_interaction_matrix, estimate_phase_pdf

"""
Generate a random, symmetric interaction matrix χ and return it. 

# Arguments 
- `N::Integer`: The number of components. 
- `μ::Real`: The mean of the values.  
- `σ::Real`: The standard deviation of the values. 

# Returns: 
The random interaction matrix χ. 
"""
function generate_random_interaction_matrix(N::Integer, μ::Real, σ::Real)::Matrix
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
- `num_phases::Integer`: The number of phases to produce. 
- `num_comps::Integer`: The number of different compositions to make out of the phases. 

# Returns: 
A matrix of the different compositions of phases. 
"""
function generate_starting_phases(num_phases::Integer, num_comps::Integer)::Matrix
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
- `χ::Matrix`: The interaction matrix. 

# Returns 
A tuple `(μ, p)` containing the chemical potential and pressure for each phase. 
`μ` is a `Vector` and `p` is a `Real`.
"""
function calc_diffs(ϕ::Vector{Float64}, χ::Matrix)::Tuple{Vector, Real}
    ϕ_sol = 1 - sum(ϕ)

    if ϕ_sol < 0  # Sanity check. 
        error("Solvent has negative concentration: ϕ_sol=$ϕ_sol")
    end
    if any(map((x) -> x < 0.0, ϕ))  # Sanity check. 
        error("Negative component concentration: ϕ=$ϕ")
    end

    log_ϕ_sol = log(ϕ_sol)
    μ = map(log, ϕ)
    p = -1 * log_ϕ_sol
    for i in 1:length(ϕ)
        v = dot( χ[i,:], ϕ )
        μ[i] += v - log_ϕ_sol
        p += 0.5 * v * ϕ[i]
    end # for i in 1:length(ϕ)

    return μ, p

end # function calc_diffs


"""
Compute the rates of change for all of the compositions. 

# Arguments 
- `ϕ::Matrix`: The compositions of the components. 
- `χ::Matrix`: The interaction matrix for the components. 

# Returns 
The rate of change of the composition (Eq. 4) in the different phases. 
"""
function evolution_rate(ϕ::Matrix, χ::Matrix)::Matrix
    num_phases, num_comps = size(ϕ)

    # get chemical potential and pressure for all components and phases
    μ = zeros(num_phases, num_comps)
    p = zeros(num_phases)
    for i in 1:num_phases
        tμ, tp = calc_diffs(ϕ[i, :], χ)
        μ[i,:] .= tμ
        p[i] = tp 
    end # for i in 1:num_phases

    # calculate rate of change of the composition in all phases
    dc = zeros(num_phases, num_comps)
    for n in 1:num_phases
        for m in 1:num_phases
            ∇p = p[n] - p[m]
            for i in 1:num_comps
                ∇μ = μ[m, i] - μ[n, i]
                dc[n, i] += ϕ[n, i] * (ϕ[m, i] * ∇μ - ∇p)
            end # for i
        end # for m
    end # for n

    return dc

end # function evolution_rate

"""
Evolves a system with the given interactions using the given time step and number of steps. 
Repeatedly updates the system by adding (Eq. 4) multiplied by the time step. 

# Arguments: 
- `ϕ::Matrix`: The compositions of the phases. 
- `χ::Matrix`: The component interaction matrix. 
- `∇t::Real`: The time steps of the simulation. 
- `steps::Integer`: The number of simulation steps. 

# Returns: 
The new, updated ϕ.
"""
function iterate_inner(ϕ::Matrix, χ::Matrix, ∇t::Real, steps::Integer)::Matrix

    tϕ = deepcopy(ϕ)

    for _ in 1:steps 
        dϕ = ∇t * evolution_rate(tϕ, χ)
        tϕ = tϕ .+ dϕ

        # Check for valid results. 
        if any(map(isnan, tϕ))
            error("Nan result in ϕ")
        end
        if any(map((x) -> x < 0.0, tϕ))
            error("Non-positive concentrations:\n∇t=$∇t\ntϕ\n$(repr("text/plain", tϕ))\nϕ\n$(repr("text/plain", ϕ))\ndϕ\n$(repr("text/plain", dϕ))")
        end
        for i in 1:size(tϕ)[1]
            if sum(tϕ[i,:]) < 0.0
                error("Non-positive solvent concentrations:\n∇t=$∇t\ntϕ\n$(repr("text/plain", tϕ))\nϕ\n$(repr("text/plain", ϕ))\ndϕ\n$(repr("text/plain", dϕ))")
            end
        end
    end # for _ in 1:steps

    return tϕ

end # function iterate_inner

"""
Evolve the interaction dynamics of χ using the given initial conditions to its final, stable state. 

# Arguments:
- `χ::Matrix`: The interaction dynamics of the components. 
- `ϕ_initial::Matrix`: The initial consentration of the components. 
- `∇t_initial::Real = 1.0`: The initial time step. 
- `tracker_interval::Real = 10.0`: The initial number of steps to observe for when iterating each step. 
- `tolerance::Real = 1e-4`: The tolerance for determining whether or not the simulation has converged. 

# Returns: 
The final composition of all of the phases.
"""
function evolve_dynamics(
        χ::Matrix, 
        ϕ_initial::Matrix;
        ∇t_initial::Real = 1.0, 
        tracker_interval::Real = 10.0,
        tolerance::Real = 1e-4
    )::Matrix
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

    # Iterate until converged. 
    while !test_converged()

        ϕ_last = deepcopy(ϕ)

        rerun = true
        while rerun
            rerun = false 
            try 
                ϕ = iterate_inner(ϕ, χ, ∇t, steps_inner)
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
- `ϕ::Matrix`: The final state that we are counting the phases in. 

# Returns 
The number of phases present. 
"""
function count_phases(ϕ::Matrix)::Integer
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
- `χ::Matrix`: The interaction matrix we are testing. 
- `n_samples::Integer`: The number of samples to return. 

# Returns: 
An array of the samples. 
"""
function estimate_phase_pdf(χ::Matrix; n_samples::Integer=64)::Vector{Integer}
    N = size(χ, 1)
    
    n_phases = ThreadsX.collect(count_phases(evolve_dynamics(χ, generate_starting_phases(N+2, N))) for _ in 1:n_samples)

    return n_phases
end # function estimate_phase_pdf

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
        n_samples::Integer=64
    )

    p = Progress(length(μs)*length(σs))

    means = zeros(length(μs), length(σs))
    stds = zeros(length(μs), length(σs))

    for (i, μ) in enumerate(μs) 
        for (j, σ) in enumerate(σs) 
            χ = generate_random_interaction_matrix(10, μ, σ)
            samples = estimate_phase_pdf(χ, n_samples=n_samples)
            means[i,j] = mean(samples)
            stds[i,j] = std(samples)
            next!(p)
        end
    end

    return (means, stds)

end


end # module