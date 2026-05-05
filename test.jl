using Pkg
Pkg.activate(".")

using Flux
using NonlinearSolve
using Zygote
using Optimisers
using ImplicitDifferentiation
using Random
using LinearAlgebra

Random.seed!(42)

# ============================================================
# PART 1: THE NONLINEAR PROBLEM (5D)
# ============================================================
#
# A coupled system with circular structure:
# Each equation has a cubic self-term and two cross-coupling
# terms mediated by the input parameters x.
#
# This is a natural extension of the 2D system:
#   2D: y[i]^3 + x[i]*y[j] - 1 = 0
#   5D: y[i]^3 + x[i]*y[i+1] + 0.5*x[i+1]*y[i+2] - 1 = 0
#        (indices wrap around cyclically)

n_dim = 5  # Problem dimension

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
# PART 3: NEURAL NETWORK (scaled up)
# ============================================================

n_input = n_dim
n_output = n_dim

model = Chain(
    Dense(n_input, 128, relu),
    Dense(128, 128, relu),
    Dense(128, 64, relu),
    Dense(64, n_output)
)

# ============================================================
# PART 4: LOSS FUNCTION
# ============================================================

λ = 0.1

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

# More samples since the problem is harder
n_samples = 500
X_data = [rand(Float32, n_input) .- 0.5f0 for _ in 1:n_samples]

n_val = 100
X_val = [rand(Float32, n_input) .- 0.5f0 for _ in 1:n_val]

opt = Adam(1e-3)
opt_state = Optimisers.setup(opt, model)

function train!(model, opt_state, X_data; n_epochs=800, batch_size=32)
    n_samples = length(X_data)

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

        if epoch % 50 == 1 || epoch == n_epochs
            avg_loss = epoch_loss / n_batches
            println("Epoch $(lpad(epoch, 4))/$(n_epochs) | Loss: $(round(avg_loss; digits=6))")
        end
    end

    return model, opt_state
end

n_epochs = 800
batch_size = 32

println("Training on $(n_dim)D nonlinear system...")
println("Network: $(n_input) → 128 → 128 → 64 → $(n_output)")
println("Training samples: $(n_samples), Validation: $(n_val)")
println()

model, opt_state = train!(
    model, opt_state, X_data;
    n_epochs=n_epochs, batch_size=batch_size
)

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
cold_start = ones(Float64, n_dim)  # [1.0, 1.0, 1.0, 1.0, 1.0]

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
println("\n" * "-"^60)
println("AGGREGATE METRICS (averaged over $(n_val) samples)")
println("-"^60)
println("  Avg ||phi(x, y_hat)||² (residual) = $(round(total_residual_norm / n_val; digits=8))")
println("  Avg MSE(y_hat, y*)                = $(round(total_mse / n_val; digits=8))")
println("  Avg Loss                          = $(round(total_loss / n_val; digits=8))")

println("\n" * "-"^60)
println("ITERATION COUNT: NN warm start vs cold start")
println("-"^60)
println("  Avg iterations (cold start) = $(round(cold_iters_total / n_val; digits=2))")
println("  Avg iterations (NN start)   = $(round(nn_iters_total / n_val; digits=2))")
println("  Iteration savings           = $(round((1 - nn_iters_total/cold_iters_total) * 100; digits=1))%")
