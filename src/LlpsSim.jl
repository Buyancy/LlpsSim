module LlpsSim

using Random, Distributions, LinearAlgebra, Distances, Clustering, ThreadsX, MultivariateStats, Plots, ProgressMeter, Memoize
Random.seed!(19)

export compute_phases, generate_random_interaction_matrix, estimate_phase_pdf

"""
# Arguments 
- `N::Integer`: The number of components. 
- `μ::Real`: The mean of the values.  
- `σ::Real`: The standard deviation of the values. 

Generate a random, symmetric interaction matrix χ and return it. 
"""
function generate_random_interaction_matrix(N::Integer, μ::Real, σ::Real)
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
```generate_starting_phases(num_phases::Integer, num_comps::Integer)```

Generate and return a starting condition for the phases. 

# Arguments: 
- num_phases::Integer: The number of phases to produce. 
- num_comps::Integer: The number of different compositions to make out of the phases. 

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
```calc_diffs(ϕ::Vector{Float64}, χ::Matrix)```

# Arguments: 
- ϕ::Vector{Float64}: The fractions of the phases. 
- χ::Matrix: The interaction matrix. 

# Returns 
A tuple (μ, p) containing the chemical potential and pressure for each phase. 
"""
function calc_diffs(ϕ::Vector{Float64}, χ::Matrix)
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
- ϕ::Matrix: The compositions of the components. 
- χ::Matrix: The interaction matrix for the components. 

# Returns 
The rate of change of the composition (Eq. 4)
"""
function evolution_rate(ϕ::Matrix, χ::Matrix)
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
```iterate_inner(ϕ::Matrix, χ::Matrix, ∇t::Real, steps::Integer)```

Evolves a system with the given interactions. 

# Arguments: 
- ϕ::Matrix: The compositions of the phases. 
- χ::Matrix: The component interaction matrix. 
- ∇t::Real: The time steps of the simulation. 
- steps::Integer: The number of simulation steps. 

# Returns: 
The new ϕ.
"""
function iterate_inner(ϕ::Matrix, χ::Matrix, ∇t::Real, steps::Integer)

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
```
evolve_dynamics(
    χ::Matrix, 
    ϕ_initial::Matrix;
    ∇t_initial::Real = 1.0, 
    tracker_interval::Real = 10.0,
    tolerance::Real = 1e-4
)::Matrix
```

Evolve the interaction dynamics of χ using the given initial conditions. 

# Arguments:
- χ::Matrix: The interaction dynamics of the components. 
- ϕ_initial::Matrix: The initial consentration of the components. 

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
```count_phases(ϕ::Matrix)```

Count the number of phases in the final stable state of ϕ. 

# Arguments 
- ϕ::Matrix: The final state that we are counting the phases in. 

# Returns 
The number of phases present. 
"""
function count_phases(ϕ::Matrix)::Integer
    # Compute the pairwise distnaces between the phases. 
    # ϕ is indexed (phases, comps)
    dists = pairwise(Euclidean(), transpose(ϕ), dims=2)
    # display(dists)

    # markov_cluster_result = mcl(dists)
    # return nclusters(markov_cluster_result)

    hc = hclust(dists, linkage=:ward)

    cut = cutree(hc, h=1e-2)
    # cut = cutree(hc, h=1.0)

    # println(cut)
    # println(unique(cut))
    # println(length(unique(cut)))

    return maximum(cut)

end # function count_phases


"""
Compute the estimated distribution on the number of phases that 
a given interactoin matrix can produce. 

# Returns: 
Tuple (P, μ, σ) of estimating the probability distribution of the number of phases χ can produce. 
- `P`: A function Integer -> Real that represents the probability of getting a 
    given number of phases from a given interaction matrix χ. 
- `μ`: The mean of P. 
- `σ`: The standard deviation of P. 
"""
function estimate_phase_pdf(χ::Matrix; n_samples::Integer=64)
    N = size(χ, 1)
    
    n_phases = ThreadsX.collect(count_phases(evolve_dynamics(χ, generate_starting_phases(N+2, N))) for _ in 1:n_samples)

    @memoize function P(k) 
        return count(==(k), n_phases) / length(n_phases)
    end

    μ = mean(n_phases)
    σ = std(n_phases)

    return (P, μ, σ)
end # function estimate_phase_pdf

"""
Generate and return the description matrices. 

# Arguments: 
- `μs::Vector{Float64}`: The μ values to test. 
- `σs::Vector{Float64}`: The σ values to test. 

# Returns: 
A tuple of (mean, stds) where each one is a matrix containing the 
statistics for the given values in it. 
"""
function generate_discription_matrices(μs::Vector{Float64}, σs::Vector{Float64})

    p = Progress(length(μs)*length(σs))

    means = zeros(length(μs), length(σs))
    stds = zeros(length(μs), length(σs))

    for (i, μ) in enumerate(μs) 
        for (j, σ) in enumerate(σs) 
            χ = generate_random_interaction_matrix(10, μ, σ)
            P, mean, std = estimate_phase_pdf(χ, n_samples=25)
            means[i,j] = mean
            stds[i,j] = std
            next!(p)
        end
    end

    return (means, stds)

end


end # module