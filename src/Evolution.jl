module Evolution

using LlpsSim, Gen, Statistics, ProgressMeter, Random, Distributions, Memoize

"""
Generate a matrix from a trace. 
"""
function matrix_from_trace(trace, N::Integer)::Matrix
    χ = zeros(N,N)
    for i in 1:N
        for j in (i+1):N
            χ[i,j] = Gen.get_choices(trace)[:χ => i => j]
            χ[j,i] = Gen.get_choices(trace)[:χ => i => j]
        end
    end
    return χ
end

"""
A function to generate an interaction matrix for `N` components drawn from 
a normal distribution with mean `μ` and standard deviation `σ`^2. 
"""
@gen function generate_normal_interaction_matrix(N::Integer, μ::Real, σ::Real)
    χ = zeros(N, N)
    for i in 1:N
        for j in (i+1):N
            χ[i,j] = {i => j} ~ normal(μ, σ) 
            χ[j,i] = χ[i,j]
        end
    end
    return χ
end

"""
A function to generate an interaction matrix for `N` components drawn from 
a uniform distribution with min `min` and max `max`. 
"""
@gen function generate_uniform_interaction_matrix(N::Integer, min::Real, max::Real)
    χ = zeros(N, N)
    for i in 1:N
        for j in (i+1):N
            χ[i,j] = {i => j} ~ uniform(min, max) 
            χ[j,i] = χ[i,j]
        end
    end
    return χ
end

"""
Uses generative modeling (elaborate) to identify interaction matrices that optimize the 
objective function. 
"""
function identify_matrix(N::Integer, objective::Function, target::Real)

    # The model that we will use to generate and score our function. 
    @gen function interaction_model() 

        # Build our interaction matrix. 
        # χ_μ = {:χ_μ} ~ uniform(0, 8)
        # χ_σ = {:χ_σ} ~ uniform(0, 8)
        χ = {:χ} ~ generate_uniform_interaction_matrix(N, -2, 5)

        # Get the number of phases in the matrix. 
        # This may not be nescicary given the objective function. 
        objective_value = objective(χ)

        {:objective} ~ normal(target, 0.1)
        
        return χ

    end # function interaction_model

    # How we will update the model at each iteration of the simulation. 
    function resimulation_update(trace)
        # Update the matrix entries. 
        for i in 1:N
            for j in (i+1):N
                trace, accept = mh(trace, select(:χ => i => j))
            end
        end

        return trace

    end # function resimulation_update

    # Generate the choicemap that holds what we are optimizing for. 
    targets = Gen.choicemap()
    targets[:objective] = target

    scores = []

    trace, weight = generate(interaction_model, (), targets)
    # for _ in 1:100
    #     trace = resimulation_update(trace)
    #     m = mean(LlpsSim.estimate_phase_pdf(Gen.get_retval(trace), n_samples=10))
    #     push!(scores, m)
    #     display(m)
    # end

    @gen function matrix_proposal(trace)
        for i in 1:N
            for j in (i+1):N
                {:χ => i => j} ~ normal(trace[:χ => i => j], 0.1)
            end
        end
    end

    function gaussian_drift_update(trace) 
        # The drift update. 
        trace, _ = mh(trace, matrix_proposal, ())

        return trace
    end

    # trace = simulate(interaction_model, ())
    # display(Gen.get_choices(trace))
    # trace, _ = Gen.importance_resampling(interaction_model, (), targets, 300, verbose=true)

    for i in 1:300
        trace = gaussian_drift_update(trace)
    end

    display(Gen.get_choices(trace))
    # display(scores)
    display(Gen.get_retval(trace))
    display(mean(LlpsSim.estimate_phase_pdf(Gen.get_retval(trace), n_samples=25)))

end # function identify_matrix_evolution



"""
A function that will return a function that is the same as in the paper (Eq. 5)
based on the parameters given. 

# Arguments: 
- `target_phase_number::Int`: The target number of phases we wish to hit. 
- `w::Real`: The penalty factor. (See paper.)
- `n_samples::Integer=32`: The number of samples to use for estimating the phase number distribution.

# Returns: 
A function that takes a `Matrix` and will return a `Real` which computes the objective function. 
"""
function target_phase_objective_function(
    target_phase_number::Int, 
    w::Real; 
    n_samples::Integer=32
)

    # The function that will do the computation. 
    @memoize function g(χ::Matrix)::Real
        N = size(χ, 1)
        K_MAX = N+2 
        samples = LlpsSim.estimate_phase_pdf(χ, n_samples=n_samples)
        P(k) = count((x) -> x == k, samples) / length(samples)
        return sum([ P(K) * exp(-(K-target_phase_number)^2 / (2 * w^2)) for K in 1:K_MAX])
    end # function g

    return g

end


"""
Replicate the evolutionary algorithm in the paper to optimize a specific objective function. 

# Arguments: 
- `objective_function`: The function that the algorithm will try to maximize in each 
    generation. The function should take in a `Matrix` χ representing the interaction matrix
    and return a `Real` which would be the score. 
- `POPULATION_SIZE::Integer=32`: The number of matrices in the "population" of the algorithm. 
- `N::Integer=5`: The number of components in the simulation. 
- `ITERATIONS::Integer=100`: The number of "generations" to use in the algotithm. 
- `RETURN_INTERMEDIATES::Bool=false`: Whether to return intermediate values in the simulation.
- `χ_BOUND::Real=10.0`: The threshold to use for when we renormalize the matrices. 

# Returns: 
Will return different things depending on the value of `RETURN_INTERMEDIATES`. 
If `RETURN_INTERMEDIATES` is false, then it will just return the optimal interaction `Matrix`
that was found through the evolutionary algorithm. 
If `RETURN_INTERMEDIATES` is true, then it will return a tuple of `(Vector{Matrix}, Vector{Real}, Matrix)`
where the first two elements contain the best matrix and score from each generation respectively
and the final element is the best overall matrix obtained by the algorithm. 
"""
function evolutionary_algorithm(
    objective_function;
    POPULATION_SIZE::Integer=32, 
    N::Integer=5, 
    ITERATIONS::Integer=100, 
    RETURN_INTERMEDIATES::Bool=false, 
    χ_BOUND::Real=10.0
) 

    # The "population" of the matrices that will be evolved over time. 
    matrices = Vector{Matrix}()
    for _ in 1:POPULATION_SIZE
        χ = generate_uniform_interaction_matrix(N, -2, 5)
        push!(matrices, χ)
    end # for _ in 1:POPULATION_SIZE

    println("Finished generating initial matrix population.")


    # A function that mutates a single matrix and returns it. 
    function mutate_matrix(χ::Matrix)::Matrix 
        χ_prime = zeros(size(χ))
        for i in 1:N
            for j in (i+1):N
                χ_prime[i,j] = rand(Normal(χ[i,j], 0.1)) # Standard gaussian drift. 
                χ_prime[j,i] = χ_prime[i,j]
            end
        end

        # Renormalize so that the values don't "blow up". 
        if mean(χ) > χ_BOUND 
            m = mean(χ)
            χ = map((x) -> x * (χ_BOUND / m), χ)
        end

        return χ_prime
        
    end # function mutate_matrix

    if RETURN_INTERMEDIATES
        intermediate_matrices = Vector{Matrix}()
        intermediate_scores = Vector{Real}()
    end


    # Iteratively improve the population. 
    @showprogress 1 "Evolving matrices..." for i in 1:ITERATIONS
        # Mutate
        matrices = map(mutate_matrix, matrices)

        # Sort the matrices by the scoring function. 
        # scores = get_score_dict(matrices)
        sort!(matrices, by=objective_function) # Sort the matrices by the objective function. 

        # Remove the bottom 30%
        num_to_kill = convert(Integer, floor(length(matrices) * 0.3))
        for j in 1:num_to_kill
            popfirst!(matrices) # Remove the ones with the lowest objective function. 
        end

        # Select random good ones to replace. 
        reverse!(matrices) # But the good ones at the front. 
        for j in 1:num_to_kill
            push!(matrices, deepcopy(matrices[j]))
        end

        if RETURN_INTERMEDIATES
            push!(intermediate_matrices,                    first(matrices))
            push!(intermediate_scores  , objective_function(first(matrices)))
        end

    end # i in 1:ITERATIONS

    # scores = get_score_dict(matrices)
    sort!(matrices, by=objective_function)

    if RETURN_INTERMEDIATES
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
- `w::Real`: The penalty factor. (See paper.)
- `n_samples::Integer=32`: The number of samples to use for estimating the phase number distribution.

# Returns: 
A function that takes a `Matrix` and will return a `Real` which computes the objective function. 
"""
function two_phase_objective_function(
    first_target_phase_number::Int, 
    second_target_phase_number::Int,
    w::Real; 
    n_samples::Integer=32
)

    first_objective = target_phase_objective_function(first_target_phase_number, w, n_samples=n_samples)
    second_objective = target_phase_objective_function(second_target_phase_number, w, n_samples=n_samples)

    @memoize function objective(χ::Matrix)::Real
        χ = deepcopy(χ)
        a = first_objective(χ)
        χ[1,2] = 0.0
        χ[2,1] = 0.0
        b = second_objective(χ)
        return mean([a, b])
    end

    return objective 

end # function two_phase_objective_function



end #module Evolution