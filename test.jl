using Pkg
Pkg.activate(".")

using Flux
using NonlinearSolve
using Zygote
using Optimisers
using ImplicitDifferentiation
using Random
using LinearAlgebra

"""
    run_warm_start_experiment(;
        n_dim=5,
        hidden_dim=128,
        n_samples=500,
        n_val=100,
        n_epochs=800,
        batch_size=32,
        λ=0.1,
        lr=1e-3,
        seed=42
    )

Train a neural network to warm-start a nonlinear solver on a
coupled n_dim-dimensional system with circular structure.

Returns a named tuple with results.
"""
function run_warm_start_experiment(;
    n_dim=5,
    hidden_dim=128,
    n_samples=500,
    n_val=100,
    n_epochs=800,
    batch_size=32,
    λ=0.1,
    lr=1e-3,
    seed=42
)
    Random.seed!(seed)

    # ============================================================
    # PART 1: THE NONLINEAR PROBLEM
    # ============================================================

    function phi(x, y)
        return [
            y[i]^3 + x[i] * y[mod1(i + 1, n_dim)] + 0.5 * x[mod1(i + 1, n_dim)] * y[mod1(i + 2, n_dim)] - 1.0
            for i in 1:n_dim
        ]
    end

    # ============================================================
    # PART 2: WRAP THE SOLVER FOR IFT
    # ============================================================

    function forward_solver(p)
        x = p[1:n_dim]
        y0 = p[n_dim+1:2*n_dim]

        residual!(u, params) = phi(params, u)
        prob = NonlinearProblem(residual!, Float64.(y0), Float64.(x))
        sol = solve(prob, NewtonRaphson(); abstol=1e-10)
        return (sol.u, nothing)
    end

    function forward_solver_with_iters(p)
        x = p[1:n_dim]
        y0 = p[n_dim+1:2*n_dim]

        residual!(u, params) = phi(params, u)
        prob = NonlinearProblem(residual!, Float64.(y0), Float64.(x))
        sol = solve(prob, NewtonRaphson(); abstol=1e-10)

        n_iters = sol.stats.nsteps
        return (sol.u, n_iters)
    end

    function conditions(p, y, z)
        x = p[1:n_dim]
        return phi(x, y)
    end

    implicit_solver = ImplicitFunction(forward_solver, conditions)

    # ============================================================
    # PART 3: NEURAL NETWORK
    # ============================================================

    model = Chain(
        Dense(n_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim ÷ 2, relu),
        Dense(hidden_dim ÷ 2, n_dim)
    )

    # ============================================================
    # PART 4: LOSS FUNCTION
    # ============================================================

    function loss_single(m, x)
        y_hat = m(x)

        # Term 1: Physics residual
        residual = phi(x, y_hat)
        physics_loss = sum(residual .^ 2)

        # Term 2: Fixed-point loss (solver in the loop via IFT)
        p = vcat(x, y_hat)
        (y_star, _) = implicit_solver(p)
        supervised_loss = sum((y_hat .- y_star) .^ 2)

        return physics_loss + λ * supervised_loss
    end

    # ============================================================
    # PART 5: TRAINING
    # ============================================================

    X_data = [rand(Float32, n_dim) .- 0.5f0 for _ in 1:n_samples]
    X_val = [rand(Float32, n_dim) .- 0.5f0 for _ in 1:n_val]

    opt = Adam(lr)
    opt_state = Optimisers.setup(opt, model)

    println("="^60)
    println("EXPERIMENT CONFIGURATION")
    println("="^60)
    println("  Problem dimension:  $n_dim")
    println("  Hidden dimension:   $hidden_dim")
    println("  Network:            $n_dim → $hidden_dim → $hidden_dim → $(hidden_dim ÷ 2) → $n_dim")
    println("  Training samples:   $n_samples")
    println("  Validation samples: $n_val")
    println("  Epochs:             $n_epochs")
    println("  Batch size:         $batch_size")
    println("  λ (loss weight):    $λ")
    println("  Learning rate:      $lr")
    println("="^60)
    println()

    training_losses = Float64[]

    for epoch in 1:n_epochs
        perm = randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            X_batch = X_data[perm[batch_start:batch_end]]

            loss_val, grads = Zygote.withgradient(model) do m
                total = 0.0
                for x in X_batch
                    total += loss_single(m, x)
                end
                total / length(X_batch)
            end

            opt_state, model = Optimisers.update(opt_state, model, grads[1])

            epoch_loss += loss_val
            n_batches += 1
        end

        avg_loss = epoch_loss / n_batches
        push!(training_losses, avg_loss)

        if epoch % 50 == 1 || epoch == n_epochs
            println("Epoch $(lpad(epoch, 4))/$(n_epochs) | Loss: $(round(avg_loss; digits=6))")
        end
    end

    # ============================================================
    # PART 6: EVALUATION
    # ============================================================

    println("\n" * "="^60)
    println("EVALUATION ($(n_dim)D System)")
    println("="^60)

    total_residual_norm = 0.0
    total_mse = 0.0
    total_loss = 0.0
    cold_iters_total = 0
    nn_iters_total = 0
    cold_start = ones(Float64, n_dim)

    for i in 1:n_val
        x = X_val[i]
        y_hat = model(x)

        # Solve with NN warm start
        _, nn_iters = forward_solver_with_iters(vcat(Float64.(x), Float64.(y_hat)))

        # Solve with cold start
        _, cold_iters = forward_solver_with_iters(vcat(Float64.(x), cold_start))

        # Solver solution using nn prediction as warm start
        y_star, _ = forward_solver(vcat(Float64.(x), Float64.(y_hat)))

        # Metrics
        residual = phi(x, y_hat)
        residual_norm = sum(residual .^ 2)
        mse = sum((y_hat .- y_star) .^ 2)
        loss = residual_norm + λ * mse

        total_residual_norm += residual_norm
        total_mse += mse
        total_loss += loss
        nn_iters_total += nn_iters
        cold_iters_total += cold_iters

        # Print details for first 5 samples
        if i <= 5
            println("\nSample $i:")
            println("  x              = $(round.(x; digits=4))")
            println("  y_hat  (nn)    = $(round.(y_hat; digits=6))")
            println("  y_star (solver)= $(round.(y_star; digits=6))")
            println("  ||phi(x, y_hat)||² = $(round(residual_norm; digits=8))")
            println("  MSE(y_hat, y*)     = $(round(mse; digits=8))")
            println("  Loss               = $(round(loss; digits=8))")
            println("  Iterations: cold=$(cold_iters), nn=$(nn_iters)")
        end
    end

    # Aggregate metrics
    avg_residual = total_residual_norm / n_val
    avg_mse = total_mse / n_val
    avg_loss = total_loss / n_val
    avg_cold_iters = cold_iters_total / n_val
    avg_nn_iters = nn_iters_total / n_val
    iter_savings = (1 - nn_iters_total / cold_iters_total) * 100

    println("\n" * "-"^60)
    println("AGGREGATE METRICS (averaged over $(n_val) samples)")
    println("-"^60)
    println("  Avg ||phi(x, y_hat)||² (residual) = $(round(avg_residual; digits=8))")
    println("  Avg MSE(y_hat, y*)                = $(round(avg_mse; digits=8))")
    println("  Avg Loss                          = $(round(avg_loss; digits=8))")

    println("\n" * "-"^60)
    println("ITERATION COUNT: NN warm start vs cold start")
    println("-"^60)
    println("  Avg iterations (cold start) = $(round(avg_cold_iters; digits=2))")
    println("  Avg iterations (NN start)   = $(round(avg_nn_iters; digits=2))")
    println("  Iteration savings           = $(round(iter_savings; digits=1))%")

    # Return results as a named tuple
    return (
        model = model,
        training_losses = training_losses,
        avg_residual = avg_residual,
        avg_mse = avg_mse,
        avg_loss = avg_loss,
        avg_cold_iters = avg_cold_iters,
        avg_nn_iters = avg_nn_iters,
        iter_savings = iter_savings,
        X_val = X_val,
    )
end