module Evolution

using LlpsSim, Statistics, ProgressMeter, Random, Distributions, Memoize


function generate_uniform_interaction_matrix(N, μ, σ)
    χ = zeros(N,N)
    for i ∈ 1:N 
        for j ∈ i:N
            χ[i,j] = (randn() * σ) + μ
            χ[j,i] = χ[i, j]
        end # for j ∈ i..N
    end # i ∈ 1..N 
    return χ
end # function generate_uniform_interaction_matrix

"""
A function that will return a function that is the same as in the paper (Eq. 5)
based on the parameters given. 

# Arguments: 
- `target_phase_number::Int`: The target number of phases we wish to hit. 
- `w::Float64`: The penalty factor. (See paper.)
- `n_samples::Int64=32`: The number of samples to use for estimating the phase number distribution.

# Returns: 
A function that takes a `Matrix{Float64}` and will return a `Float64` which computes the objective function. 
"""
function target_phase_objective_function(
    target_phase_number::Int, 
    w::Float64; 
    n_samples::Int64=32
)

    # The function that will do the computation. 
    @memoize function g(χ::Matrix{Float64})::Float64
        N = size(χ, 1)
        K_MAX = N+2 
        samples = LlpsSim.sample_phase_counts(χ, n_samples=n_samples)
        P(k) = count((x) -> x == k, samples) / length(samples)
        return sum([ P(K) * exp(-(K-target_phase_number)^2 / (2 * w^2)) for K in 1:K_MAX])
    end # function g

    return g

end


"""
Replicate the evolutionary algorithm in the paper to optimize a specific objective function. 

# Arguments: 
- `objective_function`: The function that the algorithm will try to maximize in each 
    generation. The function should take in a `Matrix{Float64}` χ representing the interaction matrix
    and return a `Float64` which would be the score. 
- `POPULATION_SIZE::Int64=32`: The number of matrices in the "population" of the algorithm. 
- `N::Int64=5`: The number of components in the simulation. 
- `ITERATIONS::Int64=100`: The number of "generations" to use in the algotithm. 
- `RETURN_INTERMEDIATES::Symbol=:None`: Whether to return intermediate values in the simulation could be one of `:None`, `:Best`, or `:All`.
- `χ_BOUND::Float64=10.0`: The threshold to use for when we renormalize the matrices. 

# Returns: 
Will return different things depending on the value of `RETURN_INTERMEDIATES`. 
If `RETURN_INTERMEDIATES` is `:None`, then it will just return the optimal interaction `Matrix{Float64}`
that was found through the evolutionary algorithm. 
If `RETURN_INTERMEDIATES` is `:Best`, then it will return a tuple of `(Vector{Matrix{Float64}}, Vector{Float64}, Matrix{Float64})`
where the first two elements contain the best matrix and score from each generation respectively
and the final element is the best overall matrix obtained by the algorithm. 
If `RETURN_INTERMEDIATES` is `:All`, then it will return a tuple of `(Vector{Vector{Matrix{Float64}}}, Vector{Vector{Float64}}, Matrix{Float64})`
where the first two elements contain the 
"""
function evolutionary_algorithm(
    objective_function;
    POPULATION_SIZE::Int64=32, 
    N::Int64=5, 
    ITERATIONS::Int64=100, 
    RETURN_INTERMEDIATES::Symbol=:None, 
    χ_BOUND::Float64=10.0
) 

    # The "population" of the matrices that will be evolved over time. 
    matrices = Vector{Matrix{Float64}}()
    for _ in 1:POPULATION_SIZE
        χ = generate_uniform_interaction_matrix(N, -2.0, 5.0)
        push!(matrices, χ)
    end # for _ in 1:POPULATION_SIZE

    # println("Finished generating initial matrix population.")

    # A function that mutates a single matrix in place. 
    function mutate_matrix(χ::Matrix{Float64})::Matrix{Float64} 
        for i in 1:N
            for j in (i+1):N
                χ[i,j] = rand(Normal(χ[i,j], 0.5)) # Standard gaussian drift. 
                χ[j,i] = χ[i,j]
            end
        end

        # Renormalize so that the values don't "blow up". 
        if mean(χ) > χ_BOUND 
            m = mean(χ)
            for i in 1:N
                for j in 1:N
                    χ[i,j] = χ[i,j] * (χ_BOUND / m)
                end
            end
        end

        return χ
        
    end # function mutate_matrix

    # Initialize return vectors if we need to. 
    if RETURN_INTERMEDIATES == :Best
        intermediate_matrices = Vector{Matrix{Float64}}()
        intermediate_scores = Vector{Float64}()
    elseif RETURN_INTERMEDIATES == :All
        intermediate_matrices = Vector{Vector{Matrix{Float64}}}()
        intermediate_scores = Vector{Vector{Float64}}()
    end

    # println("Initialized return buffers.")

    # Iteratively improve the population. 
    @showprogress 1 "Evolving matrices..." for i in 1:ITERATIONS
        # Mutate
        # map(mutate_matrix, matrices)
        for i in 1:length(matrices)
            mutate_matrix(matrices[i])
        end

        # println("Mutated Matrices")

        # Sort the matrices by the scoring function. 
        sort!(matrices, by=objective_function) # Sort the matrices by the objective function. 

        # println("Sorted Matrices")

        # Replace the bottom 30%.
        num_to_kill = convert(Int64, floor(length(matrices) * 0.3))
        for i in 1:num_to_kill
            matrices[i] = deepcopy(matrices[end-(i-1)])
        end

        # println("Selected Matrices")

        # Save the intermediate results if we want to. 
        if RETURN_INTERMEDIATES == :Best
            push!(intermediate_matrices, last(matrices))
            push!(intermediate_scores, objective_function(last(matrices)))
        elseif RETURN_INTERMEDIATES == :All
            # TODO: verify that this is correct. 
            push!(intermediate_matrices, matrices)
            push!(intermediate_scores, map(objective_function, matrices))
        end

        # println("Saved Intermediates")

    end # i in 1:ITERATIONS

    # scores = get_score_dict(matrices)
    sort!(matrices, by=objective_function)

    if RETURN_INTERMEDIATES != :None
        return (intermediate_matrices, intermediate_scores, last(matrices))
    else
        return last(matrices)
    end

end # function evolutionary_algorithm


"""
An objective function that will optimmize for two different phase counts when a specific 
interaction is "knocked out" (made to be zero in the matrix).

# Arguments: 
- `first_target_phase_number::Int`: The target number of phases we wish to hit without the "knockout". 
- `second_target_phase_number::Int`: The target number of phases we wish to hit with the "knockout". 
- `w::Float64`: The penalty factor. (See paper.)
- `n_samples::Int64=32`: The number of samples to use for estimating the phase number distribution.

# Returns: 
A function that takes a `Matrix{Float64}` and will return a `Float64` which computes the objective function. 
"""
function two_phase_objective_function(
    first_target_phase_number::Int, 
    second_target_phase_number::Int,
    w::Float64; 
    n_samples::Int64=32
)   
    first_obj = target_phase_objective_function(first_target_phase_number, w, n_samples=n_samples)
    second_obj = target_phase_objective_function(second_target_phase_number, w, n_samples=n_samples)

    @memoize function objective(χ::Matrix{Float64})::Float64
        χ1 = χ
        χ2 = deepcopy(χ)
        
        # Knockout all of the interactions involving the first phase. 
        N = size(χ, 1)
        for i in 1:N
            χ2[1,i] = 0
            χ2[i,1] = 0
        end

        return 0.5 * ( first_obj(χ1) + second_obj(χ2) )
    end

    return objective 

end # function two_phase_objective_function



end #module Evolution