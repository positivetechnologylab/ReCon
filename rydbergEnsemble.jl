local_or_globalLocal = "LocalGlobal"

# Command Line arguments
datasetName = ARGS[1] #either MNIST or FashionMNIST
const classNum = parse(Int64, ARGS[2]) #define which class to train for
rabiPulseName, detunePulseName = ARGS[3:4] # Rabi pulse, detune pulse specified in pulseLibrary.jl

trialName = datasetName*"_"* string(classNum) *"_output"

isdir(trialName) ? nothing : mkdir(trialName)

using MultivariateStats
using MLDatasets
using Plots; gr()

using Bloqade
using Flux
using ForwardDiff

using Random
using Printf
using Optim
using Optimization
using OptimizationOptimJL
using OptimizationPolyalgorithms
using LineSearches

using Combinatorics: combinations
using Base.Iterators: partition
using Base.Iterators: flatten
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Dates: now
using JLD2

include("utils/rydbergLearn" * local_or_globalLocal * ".jl")

hyparams = rydQGANhyperparameters(;
    numAtoms = 4,
    tSpan = [0.0 1.0],
    batchSize = 16,
    numDiscSteps = 1,
    disc_learnRate = 0.001,
    numEpochs = local_or_globalLocal == "LocalGlobal" ? [1 1 1 1 1] : [1 1 1 1],
    numLoopRepeats = 2,
)

println("rydbergLearn" * local_or_globalLocal)

open(trialName * "/hyparams.txt", "w") do io
    println(io, "rydbergLearn" * local_or_globalLocal)
    println(io, hyparams)
end

## number of atoms, tspan, and training set are all the same across ensemble members
const numPCs = (2^hyparams.numAtoms)::Int64

const PCscaleFactor = (1.0 / numPCs)::Float64 # upperbound for the size of each PC, after scaling. Value is chosen so that if each PC is maximal, they sum  to 1.

## Load data, do PCA. Training data saved in batches
if datasetName == "MNIST"
    trainData = MNIST(; Tx=Float32, split=:train, dir="MNIST")
    classNames = string.(0:9)
elseif  datasetName == "FashionMNIST"
    trainData = FashionMNIST(; Tx=Float32, split=:train, dir="FashionMNIST")
    classNames = trainData.metadata["class_names"]
else
    error("Invalid dataset choice")
end

trainData = trainData[trainData.targets .== classNum]

numObservations = length(trainData.targets)
Xtrain = reshape(trainData.features, (28*28, numObservations))

digitPCA = MultivariateStats.fit(PCA, Xtrain, pratio = 1, maxoutdim=numPCs) # NOTE, data matrix convention is different for this PCA method. Normaly, each row corresponds to a single observation. HERE, each COLUMN corresponds to a diffrent observation
@printf "PCA dimension reduction from %d to %d \n" size(digitPCA)[1] size(digitPCA)[2]

Xreduced = MultivariateStats.predict(digitPCA, Xtrain)

# need to rescale PCA features to be within generator's output domain
Xshifted = zeros(size(Xreduced))
minPCvals = zeros(numPCs)
maxPCvals = zeros(numPCs)

for i = 1:numPCs
    minPCvals[i] = minimum(Xreduced[i,:])
    maxPCvals[i] = maximum(Xreduced[i,:])

    Xshifted[i,:] = ( 1 .- (maxPCvals[i] .- Xreduced[i,:]) ./ (maxPCvals[i] - minPCvals[i]) ) .* PCscaleFactor
end

const batches = [Xshifted[:, r] for r in partition(1:numObservations, hyparams.batchSize)]::Vector{Matrix{Float64}}

println("Number of batches:", size(batches))
# println(size(batches[1]))

# define pulse shapes and parametersizations
include("utils/pulseLibrary.jl")
pulseList = [linearPulse, expPulse, sinPulse, reverseLinearPulse, trianglePulse, expDecayPulse, gaussianPulse, trapezoidPulse]
# pulseMask = BitVector(sum( map(x -> string.(pulseList) .== x, pulseNames) ) ) #selected from commandline
# pulseList = pulseList[pulseMask]
rabiPulse = pulseList[string.(pulseList) .== rabiPulseName][1]
detunePulse = pulseList[string.(pulseList) .== detunePulseName][1]

println(rabiPulse)
println(detunePulse)

# foundParams = []
# foundPositions = []


const init_positions = (rand(MersenneTwister(0), 4, 2) * 30 )::Matrix{Float64}
# const init_h = [0.9, 0.9, 0.9, 0.9] # seems to get stuck early on
# const init_h = [0.5, 0.1, 0.7, 0.3]
const init_h = [1.0, 1.0, 1.0, 1.0]

# do training

if local_or_globalLocal == "Local"
    const init_params = [1.0, -10.0, 0.0]::Vector{Float64}
    global pulse_params, positions, h_vals = rydbergLearnLocal(init_params, init_positions, init_h, rabiPulse, detunePulse, batches, hyparams)

elseif local_or_globalLocal == "LocalGlobal"
    const init_params = [1.0, -10.0, 0.0, 10]::Vector{Float64}
    global pulse_params, positions, h_vals = rydbergLearnLocalGlobal(init_params, init_positions, init_h, rabiPulse, detunePulse, batches, hyparams)
end

# push!(foundParams, pulse_params)
# push!(foundPositions, positions)

trialNum = 0

while isfile(trialName * "/" * rabiPulseName * detunePulseName * string(trialNum)*".jld2")
    global trialNum += 1
end

JLD2.jldsave(trialName * "/" * rabiPulseName * detunePulseName * string(trialNum)*".jld2", foundParams=pulse_params, foundPositions=positions, foundH=h_vals)



### generate some images from this learner

numSamplesPerLearner = 100

isdir(trialName * "_images") ? nothing : mkdir(trialName * "_images")

indiv_PCS = zeros(length(pulseList), numPCs, numSamplesPerLearner)
infer_noises = rand(MersenneTwister(10), 3, numSamplesPerLearner) * noiseUpperBound
# produce results


fake_outputs = mapslices(y -> quantumGenH(pulse_params, positions, h_vals, y, rabiPulse, detunePulse, hyparams), infer_noises, dims=(1))

fake_PCs = zeros(size(fake_outputs))

for i = 1:numPCs
    fake_PCs[i, :] = maxPCvals[i] .- ( (maxPCvals[i] - minPCvals[i]).*(1 .- (1 / PCscaleFactor) .* fake_outputs[i,:]) );
end

outImages = mapslices(x -> MultivariateStats.reconstruct(digitPCA, x) , fake_PCs, dims=(1))

# remove negative values as "non-physical"
# scale to be within [0, 1]

outImages[outImages .> 1.0] .= 1.0
outImages[outImages .< 0.0] .= 0.0
# outImages[outImages .>= 0.5] .= 1.0
# outImages[outImages .< 0.5] .= 0.0



imagePlots = Plots.Plot{}[]

for k = 1:16
    h = heatmap(reshape(outImages[:, k], 28, 28)', yflip=true, color=:grays, aspect_ratio=1, axis=([], false), legend = :none)
    # h = imshow(reshape(outImages[:, i], 28, 28))
    push!(imagePlots, h)
    # display(imagePlots[i])
end

digits_plot = plot(imagePlots... )

display(digits_plot)

png(digits_plot, trialName * "_images/" * rabiPulseName * detunePulseName * string(now()))