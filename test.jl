using Pkg
Pkg.activate(".")

using Flux          # Neural network library (like PyTorch for Julia)
using NonlinearSolve # SciML's nonlinear solver suite
using Zygote        # Automatic differentiation (Flux's backend)
using Optimisers    # Optimizer algorithms (Adam, SGD, etc.)
using ImplicitDifferentiation	# Library for IFT powered backward
using Random        # For reproducibility
using LinearAlgebra # For norm()
using Infiltrator
using Debugger

Random.seed!(42)

# ============================================================
# PART 1: THE NONLINEAR PROBLEM
# ============================================================

# n_input = 2
# n_output = 2
# 
# function phi(x, y)
#     r1 = y[1]^3 + x[1] * y[2] - 1.0
#     r2 = y[2]^3 + x[2] * y[1] - 1.0
#     return [r1, r2]
# end

n_input = 5
n_output = 5

function phi(x, y)
    # x: 5 parameters (input)
    # y: 5 variables to solve (output)

	r1 = y[1]^3 + x[1]*y[2] + 0.5*x[2]*y[3] - 1
	r2 = y[2]^3 + x[2]*y[3] + 0.5*x[3]*y[4] - 1
	r3 = y[3]^3 + x[3]*y[4] + 0.5*x[4]*y[5] - 1
	r4 = y[4]^3 + x[4]*y[5] + 0.5*x[5]*y[1] - 1
	r5 = y[5]^3 + x[5]*y[1] + 0.5*x[1]*y[2] - 1    

    return [r1, r2, r3, r4, r5]
end

# ============================================================
# PART 2: WRAP THE SOLVER FOR IFT
# ============================================================
#
# ImplicitDifferentiation.jl needs two things:
#
#   1. A "forward" function: given input p, produce output y
#      (this runs the solver)
#
#   2. A "conditions" function: F(p, y) = 0 defines the implicit equation
#      (this is just phi — the residual that equals zero at the solution)
#
# Then IFT gives us: dy/dp = -(∂F/∂y)⁻¹ (∂F/∂p)
# Zygote never touches the solver internals.

"""
Forward map: takes a parameter vector p and returns the solver solution.
p will contain BOTH x and y0 (the warm start) packed together.
"""
function forward_solver(p)
	x = p[1:n_input]
	y0 = p[n_input+1:end]

    residual!(u, params) = phi(params, u)
    prob = NonlinearProblem(residual!, Float64.(y0), Float64.(x))
    sol = solve(prob, NewtonRaphson(); abstol=1e-10)
    return (sol.u, nothing)
end

function forward_solver_with_iters(p)
	x = p[1:n_input]
	y0 = p[n_input+1:end]

    residual!(u, params) = phi(params, u)
    prob = NonlinearProblem(residual!, Float64.(y0), Float64.(x))
    sol = solve(prob, NewtonRaphson(); abstol=1e-10)

	# Calculate number of iters
	n_iters = sol.stats.nsteps

    return (sol.u, n_iters)
end

"""
Conditions: the implicit equation F(p, y) = 0 that defines the solution.
At convergence, phi(x, y*) = 0.
"""
function conditions(p, y, z)
    x = p[1:5]
    return phi(x, y)
end

# Create the implicit function: this is now a DIFFERENTIABLE solver
# ImplicitDifferentiation.jl handles the IFT automatically
implicit_solver = ImplicitFunction(forward_solver, conditions)

# ============================================================
# PART 3: NEURAL NETWORK
# ============================================================

width = 256
model = Chain(
    Dense(n_input, width, relu),
    Dense(width, width, relu),
    Dense(width, n_output)
)

# ============================================================
# PART 4: LOSS FUNCTION — SOLVER FULLY IN THE LOOP
# ============================================================
#
# L(θ) = ||phi(x, nn(x;θ))||² + λ * ||nn(x;θ) - solver(x; y0=nn(x;θ))||²
#
# Gradient flows through BOTH terms, including through the solver
# output via IFT. The computation graph looks like:
#
#                   ┌──────── phi(x, y_hat) ──► physics_loss
#                   │
#   x ──► nn(x) ──► y_hat ──────────────────────────► MSE ──► loss
#            │                                          ▲
#            │      gradient via IFT                    │
#            └──► implicit_solver([x; y_hat]) ──► y_star
#                 (differentiable!)

λ = 0.1

function loss_single(m, x)
    y_hat = m(x)

    # Term 1: Physics residual
    residual = phi(x, y_hat)
    physics_loss = sum(residual .^ 2)

    # Term 2: MSE with solver in the loop
    # This call IS differentiable — IFT computes dy_star/dp for us
	p = vcat(x, y_hat)
    (y_star, _) = implicit_solver(p)

    supervised_loss = sum((y_hat .- y_star) .^ 2)

    return physics_loss + λ * supervised_loss
end

# ============================================================
# PART 5: TRAINING
# ============================================================

n_samples = 1000
X_data = [rand(n_input) .- 0.5 for _ in 1:n_samples]

# Generate a separate validation set (unseen during training)
n_val = 50
X_val = [rand(n_input) .- 0.5 for _ in 1:n_val]

opt = Adam(1e-3)
opt_state = Optimisers.setup(opt, model)

function train!(model, opt_state, X_data; n_epochs=500, batch_size=32)
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

n_epochs=1000
batch_size=32

model, opt_state = train!(
	model, opt_state, X_data;
	n_epochs=n_epochs, batch_size=batch_size
)

# ============================================================
# PART 6: EVALUATION
# ============================================================

println("\n" * "="^50)
println("EVALUATION")
println("="^50)

for i in 1:5
    x = X_data[i]
    y_hat = model(x)
    y_star = forward_solver(vcat(x, y_hat))[1]

    println("\nSample $i:")
    println("  y_hat  (nn)     = $(round.(y_hat; digits=6))")
    println("  y_star (solver) = $(round.(y_star; digits=6))")
    println("  ||y_hat - y_star|| = $(round(norm(y_hat .- y_star); digits=8))")
    println("  phi(x, y_hat)      = $(round.(phi(x, y_hat); digits=8))")
end

total_residual_norm = 0.0
total_mse = 0.0
total_loss = 0.0
cold_iters_total = 0
nn_iters_total = 0
cold_start = ones(n_input)

for i in 1:n_val
	global total_residual_norm, total_mse, total_loss, cold_iters_total, nn_iters_total
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
println("ITERATION COUNT: NN warm start vs cold start $(cold_start)")
println("-"^60)
println("  Avg iterations (cold start) = $(round(cold_iters_total / n_val; digits=2))")
println("  Avg iterations (NN start)   = $(round(nn_iters_total / n_val; digits=2))")
println("  Iteration savings           = $(round((1 - nn_iters_total/cold_iters_total) * 100; digits=1))%")
