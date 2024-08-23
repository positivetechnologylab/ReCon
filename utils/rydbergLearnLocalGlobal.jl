
struct rydQGANhyperparameters
    numAtoms::Int64
    tSpan::Matrix{Float64}
    batchSize::Int64
    numDiscSteps::Int64
    disc_learnRate::Float64
    numEpochs::Matrix{Int64}
    numLoopRepeats::Int64
end

# convenience wrapper for initializing hyperparameters
rydQGANhyperparameters(; numAtoms = 4, tSpan = [0.0 , 1.0], batchSize = 16, numDiscSteps = 1, disc_learnRate = 0.001, numEpochs = [1 1 1 1], numLoopRepeats = 1) = rydQGANhyperparameters(numAtoms, tSpan, batchSize, numDiscSteps, disc_learnRate, numEpochs, numLoopRepeats)

noiseUpperBound = 1.0 # defines upper bound on the noise prior, to study the effects of using different distributions

#=
rydberglearn() executes the training loop for a single learner, and returns the learned
pulse parameters and atomic positions.

INPUTS
init_params: length 3 vector of initial pulse parameters, listed as [Ω, Δ, ϕ]
init_positions: N x 2 array of 2D positions; N is number of atoms
ΩpulseFunc: pulse function to be used for Ω(t). Function signature should be func(time, param, noise)
ΔpulseFunc: similar to ΩpulseFunc
batched_data: "real" data used in GAN training, presorted into batches
h_params: hyperparameters of this model

OUTPUT
final_params: learned pulse parameters
final_positions: learned positions
losses: record of training losses. Returned as a tuple (gen_losses, disc_losses)
=#

function rydbergLearnLocalGlobal(init_params, init_positions, init_h, ΩpulseFunc, ΔpulseFunc, batched_data, h_params)

    disc = Discriminator()
    opt_disc = Flux.setup(Flux.Adam(h_params.disc_learnRate), disc)
       
    current_params = copy(init_params)
    current_positions = copy(init_positions)
    current_h = copy(init_h)

    numBatches = length(batched_data)

    batched_data = shuffle(batched_data)

    for rep in 1:h_params.numLoopRepeats

        @printf "Repeat : %d \n" rep

        for epoch in 1:h_params.numEpochs[1]

            value_tracker_positions = copy(init_positions)
            counter_positions = 0

            println("layer1")
            @printf "Epoch : %d \n" epoch

            for (batchNum, batch) in enumerate(batched_data)

                @printf "Batch number: %d %d \n" batchNum numBatches

                noises = rand(3, h_params.batchSize) * noiseUpperBound

                # fake_outputs = quantumGenPositions(current_params, current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)

                fake_outputs = batchQuantumGenPositions(current_params, current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)

                # train discriminator for number of steps
                for k = 1:h_params.numDiscSteps
                    dis_loss, dis_grads = Flux.withgradient(disc) do disc
                        discriminator_loss(disc(batch), disc(fake_outputs))
                    end
                    update!(opt_disc, disc, dis_grads[1])
                end

                lossFunc(ps, d) = generator_loss(d(batchQuantumGenPositions(current_params, ps, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                # lossFunc(ps) = generator_loss(disc(batchQuantumGenPositions(current_params, ps, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                # loss2Func(ps) = generator_loss(quantumGen(current_params, ps, noises, h_params))  
                
                if batchNum < 5
                    gen_grads = @time ForwardDiff.gradient(x -> lossFunc(x, disc), current_positions)
                end

                # println(size(gen_grads))
                # println(lossFunc(current_positions))
                # @time ForwardDiff.gradient(loss2Func, current_positions)

                # @time ForwardDiff.gradient(loss2Func, current_positions)
                lower = zeros(h_params.numAtoms, 2)
                upper = ones(h_params.numAtoms, 2) .* 75.0

                optf = OptimizationFunction(lossFunc, Optimization.AutoForwardDiff())
                prob = OptimizationProblem(optf, current_positions, disc, lb=lower, ub=upper)
                result = solve(prob, OptimizationOptimJL.NelderMead(); maxiters = 5, local_maxiters = 5, show_trace = false)

                # inner_optimizer = Optim.GradientDescent()
                # result = @time optimize(lossFunc, lower, upper, current_positions, Fminbox(inner_optimizer), Optim.Options(show_trace = true, outer_iterations =1 , iterations=2); autodiff=:forward)


                # inner_optimizer = Optim.GradientDescent(;alphaguess=LineSearches.InitialStatic(alpha=0.1) , linesearch=LineSearches.Static())
                # result = optimize(lossFunc, lower, upper, current_positions, Fminbox(inner_optimizer), Optim.Options(show_trace = false, outer_iterations =1 , iterations=5); autodiff=:forward)

                # result = Optim.optimize(lossFunc,  current_positions, Adam(), Optim.Options(show_trace = true, iterations=10, time_limit=20); autodiff = :forward)

                current_positions = copy(result.minimizer)

                current_positions = max.(current_positions, 0.0+0.0001)
                current_positions = min.(current_positions, 75.0 - 0.0001)

                if batchNum % 1 ==0
                    println(current_positions)
                end

                if value_tracker_positions == current_positions
                    counter_positions +=1
                else
                    value_tracker_positions = copy(current_positions)
                    counter_positions = 0
                end

                if counter_positions > 12
                    println("No updates to positions after 12 batches! Early Stopping!")
                    break
                end

            end
        end
        
        for epoch in 1:h_params.numEpochs[2]

            value_tracker_detuning = copy(current_params)
            counter_detuning = 0

            println("layer2 global detuning")
            @printf "Epoch : %d \n" epoch
            println("current params", current_params)

            for (batchNum, batch) in enumerate(batched_data)

                @printf "Batch number: %d of %d \n" batchNum numBatches

                noises = rand(3, h_params.batchSize) * noiseUpperBound

                # fake_outputs = quantumGenDetune(current_params, current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)
                fake_outputs = batchQuantumGenGlobalDetune(current_params, current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)

                # train discriminator for number of steps
                for k = 1:h_params.numDiscSteps
                    dis_loss, dis_grads = Flux.withgradient(disc) do disc
                        discriminator_loss(disc(batch), disc(fake_outputs))
                    end
                    update!(opt_disc, disc, dis_grads[1])
                end

                # lossFuncDetune(detune) = generator_loss(disc(quantumGenDetune([current_params[1], detune, current_positions[3]] , current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                lossFuncDetune(detune, p) = generator_loss(disc(batchQuantumGenGlobalDetune([current_params[1], current_params[2], current_params[3], detune], current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                # loss2Func(ps) = generator_loss(quantumGen(current_params, ps, noises, h_params))
                
                if batchNum < 5
                    gen_grads = @time ForwardDiff.derivative(x->lossFuncDetune(x, 5.0), current_params[4])
                end

                # println(size(gen_grads))
                # println(lossFunc(current_positions))
                # @time ForwardDiff.gradient(loss2Func, current_positions)

                # @time ForwardDiff.gradient(loss2Func, current_positions)
                # lower = ones(1) .* -125.0
                # upper = ones(1) .* 125.0

                # lower =  -125.0
                # upper =  125.0
                # # inner_optimizer = Optim.GradientDescent()
                # # result = optimize(lossFuncDetune, lower, upper, [current_params[2]], Fminbox(inner_optimizer), Optim.Options(show_trace = false, outer_iterations =1 , iterations=2); autodiff=:forward)

                # result = Optim.optimize(lossFuncDetune, lower, upper, Brent(); iterations=5)
                
                lower = ones(1) .* -125.0 
                upper = ones(1) .* 125.0

                optf = OptimizationFunction(lossFuncDetune, Optimization.AutoForwardDiff())
                prob = OptimizationProblem(optf, [current_params[4]], lb=lower, ub=upper)
                result = solve(prob, OptimizationOptimJL.NelderMead(); maxiters = 5, local_maxiters = 5, show_trace = false)

                
                # result = Optim.optimize(lossFuncDetune, lower, upper, Brent(), Optim.Options(show_trace = false, iterations=5))

                # result = Optim.optimize(lossFunc,  current_positions, Adam(), Optim.Options(show_trace = true, iterations=10, time_limit=20); autodiff = :forward)

                current_params[4] = copy(result.minimizer[1])
                
                current_params[4] = max.(current_params[4],  -125.0 + 0.0001)
                current_params[4] = min.(current_params[4], 125.0  - 0.0001)
                
                println(current_params)
                # print(current_positions)

                if value_tracker_detuning == current_params
                    counter_detuning +=1
                else
                    value_tracker_detuning = copy(current_params)
                    counter_detuning = 0
                end

                if counter_detuning > 12
                    println("No updates to detuning value after 12 batches! Early Stopping!")
                    break
                end

            end
        end

        for epoch in 1:h_params.numEpochs[4]

            value_tracker_detuning = copy(current_params)
            counter_detuning = 0

            println("layer3: h")
            @printf "Epoch : %d \n" epoch
            println("current params", current_params)

            for (batchNum, batch) in enumerate(batched_data)

                @printf "Batch number: %d of %d \n" batchNum numBatches

                noises = rand(3, h_params.batchSize) * noiseUpperBound

                # fake_outputs = quantumGenDetune(current_params, current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)
                fake_outputs = batchQuantumGenH(current_params, current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)

                # train discriminator for number of steps
                for k = 1:h_params.numDiscSteps
                    dis_loss, dis_grads = Flux.withgradient(disc) do disc
                        discriminator_loss(disc(batch), disc(fake_outputs))
                    end
                    update!(opt_disc, disc, dis_grads[1])
                end

                # lossFuncDetune(detune) = generator_loss(disc(quantumGenDetune([current_params[1], detune, current_positions[3]] , current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                lossFuncH(h, p) = generator_loss(disc(batchQuantumGenH(current_params , current_positions, h, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                # loss2Func(ps) = generator_loss(quantumGen(current_params, ps, noises, h_params))
                
                # if batchNum < 5
                #     gen_grads = @time ForwardDiff.gradient(x->lossFuncH(x, 5), current_h)
                # end

                # println(size(gen_grads))
                # println(lossFunc(current_positions))
                # @time ForwardDiff.gradient(loss2Func, current_positions)

                # @time ForwardDiff.gradient(loss2Func, current_positions)
                # lower = ones(1) .* -125.0
                # upper = ones(1) .* 125.0

                # lower =  -125.0
                # upper =  125.0
                # # inner_optimizer = Optim.GradientDescent()
                # # result = optimize(lossFuncDetune, lower, upper, [current_params[2]], Fminbox(inner_optimizer), Optim.Options(show_trace = false, outer_iterations =1 , iterations=2); autodiff=:forward)

                # result = Optim.optimize(lossFuncDetune, lower, upper, Brent(); iterations=5)
                
                lower = zeros(h_params.numAtoms)
                upper = ones(h_params.numAtoms)

                optf = OptimizationFunction(lossFuncH, Optimization.AutoForwardDiff())
                prob = OptimizationProblem(optf, current_h, lb=lower, ub=upper)
                result = solve(prob, OptimizationOptimJL.NelderMead(); maxiters = 5, local_maxiters = 5, show_trace = false)

                
                # result = Optim.optimize(lossFuncDetune, lower, upper, Brent(), Optim.Options(show_trace = false, iterations=5))

                # result = Optim.optimize(lossFunc,  current_positions, Adam(), Optim.Options(show_trace = true, iterations=10, time_limit=20); autodiff = :forward)

                current_h = copy(result.minimizer)
                
                current_h = max.(current_h, 0.0 + 0.00000001)
                current_h = min.(current_h, 1.0 - 0.00000001)

                println(current_h)
                # print(current_positions)

                if value_tracker_detuning == current_params
                    counter_detuning +=1
                else
                    value_tracker_detuning = copy(current_params)
                    counter_detuning = 0
                end

                if counter_detuning > 12
                    println("No updates to detuning value after 12 batches! Early Stopping!")
                    break
                end

            end
        end

        for epoch in 1:h_params.numEpochs[3]

            value_tracker_detuning = copy(current_params)
            counter_detuning = 0

            println("layer2: local detune")
            @printf "Epoch : %d \n" epoch
            println("current params", current_params)

            for (batchNum, batch) in enumerate(batched_data)

                @printf "Batch number: %d of %d \n" batchNum numBatches

                noises = rand(3, h_params.batchSize) * noiseUpperBound

                # fake_outputs = quantumGenDetune(current_params, current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)
                fake_outputs = batchQuantumGenLocalDetune(current_params, current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)

                # train discriminator for number of steps
                for k = 1:h_params.numDiscSteps
                    dis_loss, dis_grads = Flux.withgradient(disc) do disc
                        discriminator_loss(disc(batch), disc(fake_outputs))
                    end
                    update!(opt_disc, disc, dis_grads[1])
                end

                # lossFuncDetune(detune) = generator_loss(disc(quantumGenDetune([current_params[1], detune, current_positions[3]] , current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                lossFuncDetune(detune, p) = generator_loss(disc(batchQuantumGenLocalDetune([current_params[1], detune, current_params[3], current_params[4]] , current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                # loss2Func(ps) = generator_loss(quantumGen(current_params, ps, noises, h_params))
                
                if batchNum < 5
                    gen_grads = @time ForwardDiff.derivative(x->lossFuncDetune(x, disc), current_params[2])
                end

                # println(size(gen_grads))
                # println(lossFunc(current_positions))
                # @time ForwardDiff.gradient(loss2Func, current_positions)

                # @time ForwardDiff.gradient(loss2Func, current_positions)
                # lower = ones(1) .* -125.0
                # upper = ones(1) .* 125.0

                # lower =  -125.0
                # upper =  125.0
                # # inner_optimizer = Optim.GradientDescent()
                # # result = optimize(lossFuncDetune, lower, upper, [current_params[2]], Fminbox(inner_optimizer), Optim.Options(show_trace = false, outer_iterations =1 , iterations=2); autodiff=:forward)

                # result = Optim.optimize(lossFuncDetune, lower, upper, Brent(); iterations=5)
                maxDetune = 0.0
                minDetune = -125.0

                lower = ones(1) .* minDetune
                upper = ones(1) .* maxDetune

                optf = OptimizationFunction(lossFuncDetune, Optimization.AutoForwardDiff())
                prob = OptimizationProblem(optf, [current_params[2]], lb=lower, ub=upper)
                result = solve(prob, OptimizationOptimJL.NelderMead(); maxiters = 5, local_maxiters = 5, show_trace = false)

                
                # result = Optim.optimize(lossFuncDetune, lower, upper, Brent(), Optim.Options(show_trace = false, iterations=5))

                # result = Optim.optimize(lossFunc,  current_positions, Adam(), Optim.Options(show_trace = true, iterations=10, time_limit=20); autodiff = :forward)

                current_params[2] = copy(result.minimizer[1])
                
                current_params[2] = max.(current_params[2],  minDetune + 0.0001)
                current_params[2] = min.(current_params[2], maxDetune - 0.0001)
                
                println(current_params)
                # print(current_positions)

                if value_tracker_detuning == current_params
                    counter_detuning +=1
                else
                    value_tracker_detuning = copy(current_params)
                    counter_detuning = 0
                end

                if counter_detuning > 12
                    println("No updates to local detuning value after 12 batches! Early Stopping!")
                    break
                end

            end
        end

        for epoch in 1:h_params.numEpochs[5]

            value_tracker_RF = copy(current_params)
            counter_RF = 0

            println("layer4: Rabi")
            @printf "Epoch : %d \n" epoch

            for (batchNum, batch) in enumerate(batched_data)

                @printf "Batch number: %d of %d \n" batchNum numBatches

                noises = rand(3, h_params.batchSize) * noiseUpperBound

                # fake_outputs = quantumGenRabi(current_params, current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)
                fake_outputs = batchQuantumGenRabi(current_params, current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)

                # train discriminator for number of steps
                for k = 1:h_params.numDiscSteps
                    dis_loss, dis_grads = Flux.withgradient(disc) do disc
                        discriminator_loss(disc(batch), disc(fake_outputs))
                    end
                    update!(opt_disc, disc, dis_grads[1])
                end
                
                # lossFuncRabi(rabi) = generator_loss(disc(quantumGenRabi([rabi, current_params[2], current_positions[3]] , current_positions, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                lossFuncRabi(rabi, p) = generator_loss(disc(batchQuantumGenRabi([rabi, current_params[2], current_params[3], current_params[4]] , current_positions, current_h, noises, ΩpulseFunc, ΔpulseFunc, h_params)))
                # loss2Func(ps) = generator_loss(quantumGen(current_params, ps, noises, h_params))
                
                if batchNum < 5
                    gen_grads = @time ForwardDiff.derivative(r->lossFuncRabi(r, nothing), current_params[1])
                end

                # println(lossFunc(current_positions))
                # @time ForwardDiff.gradient(loss2Func, current_positions)

                # @time ForwardDiff.gradient(loss2Func, current_positions)
                # lower = ones(1) .* 0.0
                # upper = ones(1) .* 15.0

                # lower = 0.0
                # upper = 15.0

                # # inner_optimizer = Optim.GradientDescent()
                # # result = optimize(lossFuncRabi, lower, upper, [current_params[1]], Fminbox(inner_optimizer), Optim.Options(show_trace = false, outer_iterations =1 , iterations=2); autodiff=:forward)
                # result = Optim.optimize(lossFuncRabi, lower, upper,  Brent(); iterations=5)
                # # result = Optim.optimize(lossFunc,  current_positions, Adam(), Optim.Options(show_trace = true, iterations=10, time_limit=20); autodiff = :forward)

                lower = zeros(1) .* 0.0
                upper = ones(1) .* 15.8

                optf = OptimizationFunction(lossFuncRabi, Optimization.AutoForwardDiff())
                prob = OptimizationProblem(optf, [current_params[1]], lb=lower, ub=upper)
                result = solve(prob, OptimizationOptimJL.NelderMead(); maxiters = 5, local_maxiters = 5, show_trace = false)



                current_params[1] = copy(result.minimizer[1])
                
                current_params[1] = max.(current_params[1], 0.0 + 0.001)
                current_params[1] = min.(current_params[1], 15.8 - 0.001)

                if batchNum % 1 == 0
                    println(current_params)
                end
                    # print(current_positions)

                if value_tracker_RF == current_params
                    counter_RF +=1
                else
                    value_tracker_RF = copy(current_params)
                    counter_RF = 0
                end

                if counter_RF > 12
                    println("No updates to Rabi Frequency value after 12 batches! Early Stopping!")
                    break
                end
            end
        end


    end

    return current_params, current_positions, current_h
end

disc_rng = Random.MersenneTwister(42)

function Discriminator()
    return Flux.Chain(
    Flux.Dense(2^4 => 64, relu, init=Flux.glorot_uniform(disc_rng)), 
    Flux.Dense(64 => 16, relu, init=Flux.glorot_uniform(disc_rng)), 
    Flux.Dense(16 => 1, sigmoid, init=Flux.glorot_uniform(disc_rng)), f64)
end

function quantumGenPositions(params, positions, h, noise, ΩpulseFunc, ΔpulseFunc, hyperparameters)

    # keep Ωpulse the same for now

    Ωparam, localΔparam, ϕparam, globalΔparam = collect(flatten(params))

    nsites = hyperparameters.numAtoms
    timeSpan = hyperparameters.tSpan

    atoms = [[positions[i, 1], positions[i, 2]] for i = 1:nsites]
    atoms = map(x->max.(x, 0.0), atoms)
    atoms = map(x->min.(x, 75.0), atoms)

    reg = zero_state(Complex{typeof(positions[1, 1])}, nsites)

    # Ωpulse = piecewise_linear(clocks = [0.0,  0.1, 1.0-0.1, 1.0], values = [0.0, 5.0, 5.0, 0.0])
    Ωpulse = t -> ΩpulseFunc(t, Ωparam, noise[1])

    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, timeSpan[1]+0.1, timeSpan[2]-0.1, timeSpan[2]], values = [Δ_start, Δ_start, Δparam, Δparam]) , kernel_radius = 0.1)
    Δpulse = t -> ΔpulseFunc(t, localΔparam, noise[2]) + globalΔparam

    Δmulitple = [ t -> hval * Δpulse(t) for hval in h]

    rh = rydberg_h(atoms; Δ=Δmulitple, Ω=Ωpulse, ϕ=ϕparam)
    
    problem = SchrodingerProblem(reg, timeSpan, rh)

    solve(problem, Rodas5(autodiff=false); save_everystep=false)
    
    probs = abs2.(statevec(reg))

    result = mod1.(probs, 1.0 / 2^hyperparameters.numAtoms)

    result[probs .== 0] .=0

    return result

end

function quantumGenLocalDetune(params, positions, h, noise, ΩpulseFunc, ΔpulseFunc, hyperparameters)

    # keep Ωpulse the same for now

    Ωparam, localΔparam, ϕparam, globalΔparam = collect(flatten(params))

    nsites = hyperparameters.numAtoms
    timeSpan = hyperparameters.tSpan

    atoms = [[positions[i, 1], positions[i, 2]] for i = 1:nsites]
    reg = zero_state(Complex{typeof(localΔparam)}, nsites)

    # Ωpulse = piecewise_linear(clocks = [0.0,  0.1, 1.0-0.1, 1.0], values = [0.0, 5.0, 5.0, 0.0])
    Ωpulse = t -> ΩpulseFunc(t, Ωparam, noise[1])

    Δpulse = t -> ΔpulseFunc(t, localΔparam, noise[2]) + globalΔparam

    Δmultiple = [ t -> hval * Δpulse(t) for hval in h]

    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, timeSpan[1]+0.1, timeSpan[2]-0.1, timeSpan[2]], values = [Δ_start, Δ_start, Δparam, Δparam]) , kernel_radius = 0.1)

    rh = rydberg_h(atoms; Δ=Δmultiple, Ω=Ωpulse, ϕ=ϕparam)
    
    problem = SchrodingerProblem(reg, timeSpan, rh)

    solve(problem, Rodas5(autodiff=false); save_everystep=false)
    
    probs = abs2.(statevec(reg))

    result = mod1.(probs, 1.0 / 2^hyperparameters.numAtoms)

    result[probs .== 0] .=0

    return result

end

function quantumGenGlobalDetune(params, positions, h, noise, ΩpulseFunc, ΔpulseFunc, hyperparameters)

    # keep Ωpulse the same for now

    Ωparam, localΔparam, ϕparam, globalΔparam = collect(flatten(params))

    nsites = hyperparameters.numAtoms
    timeSpan = hyperparameters.tSpan

    atoms = [[positions[i, 1], positions[i, 2]] for i = 1:nsites]
    reg = zero_state(Complex{typeof(globalΔparam)}, nsites)

    # Ωpulse = piecewise_linear(clocks = [0.0,  0.1, 1.0-0.1, 1.0], values = [0.0, 5.0, 5.0, 0.0])
    Ωpulse = t -> ΩpulseFunc(t, Ωparam, noise[1])

    Δpulse = t -> ΔpulseFunc(t, localΔparam, noise[2]) + globalΔparam

    Δmultiple = [ t -> hval * Δpulse(t) for hval in h]

    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, timeSpan[1]+0.1, timeSpan[2]-0.1, timeSpan[2]], values = [Δ_start, Δ_start, Δparam, Δparam]) , kernel_radius = 0.1)

    rh = rydberg_h(atoms; Δ=Δmultiple, Ω=Ωpulse, ϕ=ϕparam)
    
    problem = SchrodingerProblem(reg, timeSpan, rh)

    solve(problem, Rodas5(autodiff=false); save_everystep=false)
    
    probs = abs2.(statevec(reg))

    result = mod1.(probs, 1.0 / 2^hyperparameters.numAtoms)

    result[probs .== 0] .=0

    return result

end

function quantumGenRabi(params, positions, h, noise, ΩpulseFunc, ΔpulseFunc, hyperparameters)

    # keep Ωpulse the same for now

    Ωparam, localΔparam, ϕparam, globalΔparam= collect(flatten(params))

    nsites = hyperparameters.numAtoms
    timeSpan = hyperparameters.tSpan

    atoms = [[positions[i, 1], positions[i, 2]] for i = 1:nsites]
    reg = zero_state(Complex{typeof(Ωparam)}, nsites)

    # Ωpulse = piecewise_linear(clocks = [0.0,  0.1, 1.0-0.1, 1.0], values = [0.0, 5.0, 5.0, 0.0])
    Ωpulse = t -> ΩpulseFunc(t, Ωparam, noise[1])

    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, timeSpan[1]+0.1, timeSpan[2]-0.1, timeSpan[2]], values = [Δ_start, Δ_start, Δparam, Δparam]) , kernel_radius = 0.1)
    Δpulse = t -> ΔpulseFunc(t, localΔparam, noise[2]) + globalΔparam

    Δmulitple = [ t -> hval * Δpulse(t) for hval in h]

    rh = rydberg_h(atoms; Δ=Δmulitple, Ω=Ωpulse, ϕ=ϕparam)
    
    problem = SchrodingerProblem(reg, timeSpan, rh)

    solve(problem, Rodas5(autodiff=false); save_everystep=false)
    
    probs = abs2.(statevec(reg))

    result = mod1.(probs, 1.0 / 2^hyperparameters.numAtoms)

    result[probs .== 0] .=0

    return result

end

function quantumGenH(params, positions, h, noise, ΩpulseFunc, ΔpulseFunc, hyperparameters)

    # keep Ωpulse the same for now

    Ωparam, localΔparam, ϕparam, globalΔparam = collect(flatten(params))

    nsites = hyperparameters.numAtoms
    timeSpan = hyperparameters.tSpan

    atoms = [[positions[i, 1], positions[i, 2]] for i = 1:nsites]
    reg = zero_state(Complex{typeof(h[1])}, nsites)

    # Ωpulse = piecewise_linear(clocks = [0.0,  0.1, 1.0-0.1, 1.0], values = [0.0, 5.0, 5.0, 0.0])
    Ωpulse = t -> ΩpulseFunc(t, Ωparam, noise[1])

    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, timeSpan[1]+0.1, timeSpan[2]-0.1, timeSpan[2]], values = [Δ_start, Δ_start, Δparam, Δparam]) , kernel_radius = 0.1)
    Δpulse = t -> ΔpulseFunc(t, localΔparam, noise[2]) + globalΔparam

    Δmulitple = [ t -> hval * Δpulse(t) for hval in h]

    rh = rydberg_h(atoms; Δ=Δmulitple, Ω=Ωpulse, ϕ=ϕparam)
    
    problem = SchrodingerProblem(reg, timeSpan, rh)

    solve(problem, Rodas5(autodiff=false); save_everystep=false)
    
    probs = abs2.(statevec(reg))

    result = mod1.(probs, 1.0 / 2^hyperparameters.numAtoms)

    result[probs .== 0] .=0

    return result

end


batchQuantumGenPositions(params, positions, h, noises,  ΩpulseFunc, ΔpulseFunc, hyperparameters) = mapslices(x -> quantumGenPositions(params, positions, h, x, ΩpulseFunc, ΔpulseFunc, hyperparameters), noises, dims=(1))

batchQuantumGenLocalDetune(params, positions, h, noises,  ΩpulseFunc, ΔpulseFunc, hyperparameters) = mapslices(x -> quantumGenLocalDetune(params, positions, h, x, ΩpulseFunc, ΔpulseFunc, hyperparameters), noises, dims=(1))

batchQuantumGenGlobalDetune(params, positions, h, noises,  ΩpulseFunc, ΔpulseFunc, hyperparameters) = mapslices(x -> quantumGenGlobalDetune(params, positions, h, x, ΩpulseFunc, ΔpulseFunc, hyperparameters), noises, dims=(1))

batchQuantumGenRabi(params, positions, h, noises,  ΩpulseFunc, ΔpulseFunc, hyperparameters) = mapslices(x -> quantumGenRabi(params, positions, h, x, ΩpulseFunc, ΔpulseFunc, hyperparameters), noises, dims=(1))

batchQuantumGenH(params, positions, h, noises,  ΩpulseFunc, ΔpulseFunc, hyperparameters) = mapslices(x -> quantumGenH(params, positions, h, x, ΩpulseFunc, ΔpulseFunc, hyperparameters), noises, dims=(1))


function discriminator_loss(real_output, fake_output)

    real_loss = logitbinarycrossentropy(real_output, 1)
    fake_loss = logitbinarycrossentropy(fake_output, 0)
    return real_loss + fake_loss
end

function generator_loss(fake_output)

    return logitbinarycrossentropy(fake_output, 1)
end
