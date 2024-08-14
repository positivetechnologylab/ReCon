#=
library of pulses to be used for different learners in the ensemble. In general, the 
    pulse function should have a call signature pulse(t, param, noise),
        where t is time, param is the parameter to be optimized, and noise is 
        the seeding noise from latent space.
=#

# changed to be hardware compatible
function expPulse(t, param, noise)

    peakVal = param .* noise
    peakTime = 0.2

    if t <= peakTime
        val = peakVal / peakTime .* t
    else
        val = (exp(-(t- peakTime) / 0.2 ) * peakVal )
    end
    return val
end

# changed to be hardware compatible for local detuning
# actually a cosine pulse now
function sinPulse(t, param, noise)

    return (param ./ 2) .* -cos.(2 * 2*π*t) .* noise .+ (param / 2)
end

# changed to be hardware compatible
function linearPulse(t, param, noise)

    # Δ_start = -2π * 13 * noise #-2π * 13
    # Δpulse = Bloqade.smooth(piecewise_linear(clocks = [0.0, 0.1, 0.9, 1.0], values = [Δ_start, Δ_start, param, param]) , kernel_radius = 0.1)
    Δstart = 0.0
    Δmiddle = -2π * 13 * noise

    if t<=0.1
        val = (Δmiddle / 0.1) * t
    elseif t > 0.1 && t <= 0.2
        val = Δmiddle
    elseif t > 0.2 && t < 0.8
        val =  (param - Δmiddle) / (0.8 - 0.2) * (t - 0.2) + Δmiddle
    elseif t>=0.8 && t <= 0.9
        val = param
    elseif t>0.9
        val = param + (Δstart - param) / (1.0 - 0.9) * (t - 0.9)
    end

    return  val 

end

function reverseLinearPulse(t, param, noise)
    return 2π * 13 .+ (param + 2*π) / (1.0) .* t * noise
end

function trianglePulse(t, param, noise)
    
    val = 0

    amplitude = param

    if t <= 0.5
        val = amplitude / 0.5 * t
    else
        val = -amplitude / 0.5 * t + 2*param
    end
    
    return val
end

# modified to be hardware compatible on local pulses
function expDecayPulse(t, param, noise)

    peakVal = -120.0.* noise
    peakTime = 0.2

    if t <= peakTime
        val = peakVal / peakTime .* t
    else
        val = peakVal  .* exp(-(t-peakTime) * param^2)
    end

    return val
end

function gaussianPulse(t, param, noise)
    return param * exp( -(t-0.5)^2 / (0.01*noise))
end

function trapezoidPulse(t, param, noise)
    
    val = 0

    risingTime = 0.1 * noise

    if t <= risingTime
        val = param / risingTime * t
    elseif t > risingTime && t < 1.0 - risingTime
        val = param
    else
        val = - param / risingTime * (t - 1.0)
    end

    return val
end