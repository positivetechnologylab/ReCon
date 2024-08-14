using Bloqade
using Distributions: Normal, MvNormal
using Base.Iterators: flatten

system_Ω_noise = Normal(1.0, 0.01)
system_Δ_noise = Normal(0.0, 0.1)
system_X_noise = MvNormal([0.0, 0.0], 0.1 .* [1.0 0.0; 0.0 1.0])

function noisyQuantumGenGlobal(params, positions, noise, ΩpulseFunc, ΔpulseFunc, hyperparameters)

    Ωparam, Δparam, ϕparam = collect(flatten(params))

    nsites = hyperparameters.numAtoms
    timeSpan = hyperparameters.tSpan

    atoms = [[positions[i, 1], positions[i, 2] ] + rand(system_X_noise) for i = 1:nsites]

    reg = zero_state(Complex{typeof(positions[1, 1])}, nsites)

    Ωnoise = rand(system_Ω_noise)
    # Ωpulse = piecewise_linear(clocks = [0.0,  0.1, 1.0-0.1, 1.0], values = [0.0, 5.0, 5.0, 0.0])
    Ωpulse = t -> ΩpulseFunc(t, Ωparam, noise[1])  * Ωnoise

    Δnoise = rand(system_Δ_noise)
    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, timeSpan[1]+0.1, timeSpan[2]-0.1, timeSpan[2]], values = [Δ_start, Δ_start, Δparam, Δparam]) , kernel_radius = 0.1)
    Δpulse = t -> ΔpulseFunc(t, Δparam, noise[2]) + Δnoise

    rh = rydberg_h(atoms; Δ=Δpulse, Ω=Ωpulse, ϕ=ϕparam)
    
    problem = SchrodingerProblem(reg, timeSpan, rh)

    solve(problem, Rodas5(autodiff=false); save_everystep=false)
    
    probs = abs2.(statevec(reg))

    result = mod1.(probs, 1.0 / 2^hyperparameters.numAtoms)

    result[probs .== 0] .=0

    return result

end

function noisyQuantumGenGlobalLocal(params, positions, h, noise, ΩpulseFunc, ΔpulseFunc, hyperparameters)

    # keep Ωpulse the same for now

    Ωparam, localΔparam, ϕparam, globalΔparam = collect(flatten(params))

    nsites = hyperparameters.numAtoms
    timeSpan = hyperparameters.tSpan

    atoms = [[positions[i, 1], positions[i, 2]] + rand(system_X_noise) for i = 1:nsites]
    reg = zero_state(Complex{typeof(h[1])}, nsites)

    Ωnoise = rand(system_Ω_noise)
    # Ωpulse = piecewise_linear(clocks = [0.0,  0.1, 1.0-0.1, 1.0], values = [0.0, 5.0, 5.0, 0.0])
    Ωpulse = t -> ΩpulseFunc(t, Ωparam, noise[1])  * Ωnoise

    Δnoise = rand(system_Δ_noise)
    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, timeSpan[1]+0.1, timeSpan[2]-0.1, timeSpan[2]], values = [Δ_start, Δ_start, Δparam, Δparam]) , kernel_radius = 0.1)
    Δpulse = t -> ΔpulseFunc(t, localΔparam, noise[2]) + globalΔparam + Δnoise + Δnoise #add twice because noise in both global and local params

    Δmulitple = [ t -> hval * Δpulse(t) for hval in h]

    rh = rydberg_h(atoms; Δ=Δmulitple, Ω=Ωpulse, ϕ=ϕparam)
    
    problem = SchrodingerProblem(reg, timeSpan, rh)

    solve(problem, Rodas5(autodiff=false); save_everystep=false)
    
    probs = abs2.(statevec(reg))

    result = mod1.(probs, 1.0 / 2^hyperparameters.numAtoms)

    result[probs .== 0] .=0

    return result

end
