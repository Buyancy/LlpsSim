module Evolution

using LlpsSim, Gen, Statistics, ProgressMeter, Random, Distributions, Memoize

"""
Generate a matrix from a trace. 
"""
function matrix_from_trace(trace, N::Int64)::Matrix{Float64}
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
@gen function generate_normal_interaction_matrix(N::Int64, μ::Float64, σ::Float64)
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
@gen function generate_uniform_interaction_matrix(N::Int64, min::Float64, max::Float64)
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
function identify_matrix(N::Int64, objective::Function, target::Float64)

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
- `RETURN_INTERMEDIATES::Bool=false`: Whether to return intermediate values in the simulation.
- `χ_BOUND::Float64=10.0`: The threshold to use for when we renormalize the matrices. 

# Returns: 
Will return different things depending on the value of `RETURN_INTERMEDIATES`. 
If `RETURN_INTERMEDIATES` is false, then it will just return the optimal interaction `Matrix{Float64}`
that was found through the evolutionary algorithm. 
If `RETURN_INTERMEDIATES` is true, then it will return a tuple of `(Vector{Matrix{Float64}}, Vector{Float64}, Matrix{Float64})`
where the first two elements contain the best matrix and score from each generation respectively
and the final element is the best overall matrix obtained by the algorithm. 
"""
function evolutionary_algorithm(
    objective_function;
    POPULATION_SIZE::Int64=32, 
    N::Int64=5, 
    ITERATIONS::Int64=100, 
    RETURN_INTERMEDIATES::Bool=false, 
    χ_BOUND::Float64=10.0
) 

    # The "population" of the matrices that will be evolved over time. 
    matrices = Vector{Matrix{Float64}}()
    for _ in 1:POPULATION_SIZE
        χ = generate_uniform_interaction_matrix(N, -2.0, 5.0)
        push!(matrices, χ)
    end # for _ in 1:POPULATION_SIZE

    println("Finished generating initial matrix population.")


    # A function that mutates a single matrix in place. 
    function mutate_matrix(χ::Matrix{Float64})::Matrix{Float64} 
        for i in 1:N
            for j in (i+1):N
                χ[i,j] = rand(Normal(χ[i,j], 0.1)) # Standard gaussian drift. 
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

    if RETURN_INTERMEDIATES
        intermediate_matrices = Vector{Matrix{Float64}}()
        intermediate_scores = Vector{Float64}()
    end


    # Iteratively improve the population. 
    @showprogress 1 "Evolving matrices..." for i in 1:ITERATIONS
        # Mutate
        # map(mutate_matrix, matrices)
        for i in 1:N
            mutate_matrix(matrices[i])
        end

        # Sort the matrices by the scoring function. 
        sort!(matrices, by=objective_function) # Sort the matrices by the objective function. 

        # Replace the bottom 30%.
        num_to_kill = convert(Int64, floor(length(matrices) * 0.3))
        for i in 1:num_to_kill
            matrices[i] = deepcopy(matrices[end-(i-1)])
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
    @memoize function objective(χ::Matrix{Float64})::Float64
        χ1 = χ
        χ2 = deepcopy(χ)
        
        # Knockout all of the interactions involving the first phase. 
        N = size(χ, 1)
        for i in 1:N
            χ2[1,i] = 0
            χ2[i,1] = 0
        end

        # Score the two matrices.
        K_MAX = N+2 
        samples_1 = LlpsSim.sample_phase_counts(χ1, n_samples=n_samples)
        samples_2 = LlpsSim.sample_phase_counts(χ2, n_samples=n_samples)
        l1 = length(samples_1)
        l2 = length(samples_2) 
        p1 = [count((x) -> x == k, samples_1) for x in 1:K_MAX]
        p2 = [count((x) -> x == k, samples_2) for x in 1:K_MAX]
        P1(k) = p1[k] / l1
        P2(k) = p2[k] / l2
        # terms = []
        s = 0 
        for K1 in 1:K_MAX
            for K2 in 1:K_MAX
                t = exp( -((K1-first_target_phase_number)^2 * (K2-second_target_phase_number)^2) / (2*w^2) ) * P1(K1) * P2(K2)
                # push!(terms, t)
                s += t 
            end
        end
        # return sum(terms)
        return s 
    end

    return objective 

end # function two_phase_objective_function



end #module Evolution