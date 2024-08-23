classRange = 0:9
dataset = "MNIST"
dataIsSaved = true
variant = "LocalGlobal"

plotRange = 1:16

using JLD2
using MLDatasets
using MultivariateStats
using Random
using Printf
using Plots
using CSV
using Tables

using Base.Sort: PartialQuickSort
using Combinatorics: combinations

include("utils/pulseLibrary.jl")
include("utils/rydbergLearn"*variant*".jl")
include("utils/noisyGenerator.jl")
include("utils/fid.jl")


noisyQuantumGen = variant=="LocalGlobal" ? noisyQuantumGenGlobalLocal : noisyQuantumGenGlobal
idealQuantumGen = quantumGenPositions

# relevant constants
hyparams = rydQGANhyperparameters(;
    numAtoms = 4,
    tSpan = [0.0 1.0],
    batchSize = 16,
    numDiscSteps = 1,
    disc_learnRate = 0.001,
    numEpochs = [1 1 1 1 1],
    numLoopRepeats = 2,
)

numPCs = 2^hyparams.numAtoms
PCscaleFactor = 1 / numPCs # upperbound for the size of each PC, after scaling. Value is chosen so that if each PC is maximal, they sum  to 1.

function smallestn(a, n)
    sort(a; alg=Sort.PartialQuickSort(n))[1:n]
end

function genIndivImages(optParams, optPositions, trialPulses, digitPCA, maxPCvals, minPCvals; optH=nothing, plotFlag = false, noiseFlag=:noisy)

    numSamplesPerLearner = 500

    indiv_PCS = zeros(length(trialPulses), numPCs, numSamplesPerLearner)
    infer_noises = rand(MersenneTwister(10), 3, numSamplesPerLearner)

    for (j , pulseFuncSet) in enumerate(trialPulses)

        if noiseFlag == :noisy
            if variant == "LocalGlobal"
                fake_outputs = mapslices(y -> noisyQuantumGen(optParams[j], optPositions[j], optH[j],  y, pulseFuncSet[1], pulseFuncSet[2], hyparams), infer_noises, dims=(1))
            elseif variant == "Global"
                fake_outputs = mapslices(y -> noisyQuantumGen(optParams[j], optPositions[j], y, pulseFuncSet[1], pulseFuncSet[2], hyparams), infer_noises, dims=(1))
            end
        elseif noiseFlag == :ideal
            if variant == "LocalGlobal"
                fake_outputs = mapslices(y -> idealQuantumGen(optParams[j], optPositions[j], optH[j],  y, pulseFuncSet[1], pulseFuncSet[2], hyparams), infer_noises, dims=(1))
            elseif variant == "Global"
                fake_outputs = mapslices(y -> idealQuantumGen(optParams[j], optPositions[j], y, pulseFuncSet[1], pulseFuncSet[2], hyparams), infer_noises, dims=(1))
            end  
        end

        fake_PCs = zeros(size(fake_outputs))
     
        for i = 1:numPCs
             fake_PCs[i, :] = maxPCvals[i] .- ( (maxPCvals[i] - minPCvals[i]).*(1 .- (1 / PCscaleFactor) .* fake_outputs[i,:]) );
        end
    
        indiv_PCS[j, :, :] = fake_PCs
     
        outImages = mapslices(x -> MultivariateStats.reconstruct(digitPCA, x) , fake_PCs, dims=(1))
     
        # remove negative values as "non-physical"
        # scale to be within [0, 1]
        outImages[outImages .> 1.0] .= 1.0
        outImages[outImages .< 0.0] .= 0.0
        # outImages[outImages .>= 0.5] .= 1.0
        # outImages[outImages .< 0.5] .= 0.0
     
        if plotFlag == true
            imagePlots = Plots.Plot{}[]
        
            for k = plotRange
                h = heatmap(reshape(outImages[:, k], 28, 28)', yflip=true, color=:grays, aspect_ratio=1, axis=([], false), legend = :none)
                # h = imshow(reshape(outImages[:, i], 28, 28))
                push!(imagePlots, h)
                # display(imagePlots[i])
            end
        
            digits_plot = Plots.plot(imagePlots... )
        
            display(digits_plot)
            
        end
    
    end
    
    return indiv_PCS
end

function genImagesFromPC(PCs, digitPCA; plotFlag = false, plotTitle="")
    numSamplesPerLearner = 500

    outImages = mapslices(x -> MultivariateStats.reconstruct(digitPCA, x) , PCs, dims=(1))

    # remove negative values as "non-physical"
    # scale to be within [0, 1]
    outImages[outImages .> 1.0] .= 1.0
    outImages[outImages .< 0.0] .= 0.0
    # outImages[outImages .>= 0.5] .= 1.0
    # outImages[outImages .< 0.5] .= 0.0

    if plotFlag == true
        imagePlots = Plots.Plot{}[]
    
        for k = plotRange
            h = heatmap(reshape(outImages[:, k], 28, 28)', yflip=true, color=:grays, aspect_ratio=1, axis=([], false), legend = :none)
            # h = imshow(reshape(outImages[:, i], 28, 28))
            push!(imagePlots, h)
            # display(imagePlots[i])
        end
    
        digits_plot = Plots.plot(imagePlots... , plot_title=plotTitle)

        display(digits_plot)
        
    end

    return outImages
end

function greedyEnsembleFormation(generated_PCs, Xval, trialPulses, digitPCA; plotFlag = false)

    indiv_FID = zeros(length(trialPulses))

    for (j , pulseFuncSet) in enumerate(trialPulses)
            
        fake_PCs = generated_PCs[j,:,:]

        outImages = mapslices(x -> MultivariateStats.reconstruct(digitPCA, x) , fake_PCs, dims=(1))
    
        # remove negative values as "non-physical"
        # scale to be within [0, 1]
        outImages[outImages .> 1.0] .= 1.0
        outImages[outImages .< 0.0] .= 0.0
        # outImages[outImages .>= 0.5] .= 1.0
        # outImages[outImages .< 0.5] .= 0.0
    
        if plotFlag == true
            imagePlots = Plots.Plot{}[]
        
            for k = plotRange
                h = heatmap(reshape(outImages[:, k], 28, 28)', yflip=true, color=:grays, aspect_ratio=1, axis=([], false), legend = :none)
                # h = imshow(reshape(outImages[:, i], 28, 28))
                push!(imagePlots, h)
                # display(imagePlots[i])
            end
        
            digits_plot = Plots.plot(imagePlots... )
        
            display(digits_plot)
            
        end
        
        indiv_FID[j] = fid_score(outImages, Xval)
    
        # println("Val FID:", indiv_FID[j])
        
    end

    minFID = minimum(indiv_FID)

    indices = findall(x -> x == minFID, indiv_FID)

    if length(indices) == 1
        index = indices[1]
    else
        index = indices[rand(1:length(indices))]
    end

    print("Best learner is ", trialPulses[index, :])

    selectedIndices = [index]
    # ensemblePCs = generated_PCs[index, :, :]
    
    trialIndices = 1:length(indiv_FID)
    trialIndices = trialIndices[trialIndices .!= index]

    isSearching = true
    
    while isSearching

        nextIndex = nothing

        # try adding additional learners to the ensemble
        for trialIndex in trialIndices

            # println("trial", trialPulses[trialIndex, :])

            trialEnsemble = mean(generated_PCs[ [selectedIndices..., trialIndex], :, :], dims=1)[1, :, :]

            trialImages = genImagesFromPC(trialEnsemble, digitPCA)

            trial_FID = fid_score(trialImages, Xval)

            # println("trial FID", trial_FID)

            if trial_FID < minFID
                nextIndex = trialIndex
                minFID = trial_FID
            end

        end

        # choose the best result. If none improve, quit search
        if isnothing(nextIndex)
            isSearching = false
        else
            push!(selectedIndices, nextIndex)
            trialIndices = trialIndices[trialIndices .!= nextIndex]
        end
        
    end

    println("Final FID", minFID)

    return selectedIndices
end

function bruteForceEnsembleFormation(generated_PCs, Xval, trialPulses, digitPCA; plotFlag = false)

    comb = collect(combinations(1:length(trialPulses)))

    subEnsemble_digits_plot = Plots.Plot{}[]
    sub_testing_fid = zeros(length(comb))
    selectedIndices = []

    for i in 1:length(comb)

        subset_PCS = generated_PCs[comb[i], :, :]
    
        subEnsemblePCs = mean(subset_PCS, dims=1)[1, :, :]
    
        subEnsembleImages = mapslices(x -> MultivariateStats.reconstruct(digitPCA, x) , subEnsemblePCs, dims=(1))
    
        subEnsembleImages[subEnsembleImages .> 1.0] .= 1.0
        subEnsembleImages[subEnsembleImages .< 0.0] .= 0.0

        # display(subEnsemble_digits_plot)
        
        sub_testing_fid[i] = fid_score(subEnsembleImages, Xval)
    
        println(i, " / ", length(comb), "FID:", sub_testing_fid[i])
    
        # if sub_testing_fid[i] <= 40.0
        #     push!(selectedIndices, i)
        # end
    end

    numToSelect = 5
    smallestFIDS = smallestn(sub_testing_fid, numToSelect)

    selectedMask = sum( map(x -> sub_testing_fid .== x, smallestFIDS) )
    selectedMask[selectedMask .> 1] .= 1

    selectedMask = BitVector(selectedMask)

    selectedFIDs = sub_testing_fid[selectedMask]
    selectedCombs = comb[selectedMask]


    println("Brute force")
    for i in 1:numToSelect      
        println("learners", trialPulses[selectedCombs[i]],"val FID:", selectedFIDs[i])
        
    end

end

function single_ensemble_results(digitClass)

    dirName = "training_results/"*dataset*"_"*variant*"/"*dataset*"_"* string(digitClass) *"_output"

    if dataset == "MNIST"
        trainData = MNIST(; Tx=Float32, split=:train, dir="MNIST")
        classNames = string.(0:9)
    elseif  dataset == "FashionMNIST"
        trainData = FashionMNIST(; Tx=Float32, split=:train, dir="FashionMNIST")
        classNames = trainData.metadata["class_names"]
    else
        error("Invalid dataset choice")
    end

    println("Class: ", classNames[digitClass+1])
    ## Load train data, do PCA
    # trainData = MNIST(; Tx=Float32, split=:train, dir="MNIST")
    trainData = trainData[trainData.targets .== digitClass]

    numObservations = length(trainData.targets)
    Xtrain = reshape(trainData.features, (28*28, numObservations))

    digitPCA = MultivariateStats.fit(PCA, Xtrain, pratio = 1, maxoutdim=numPCs) # NOTE, data matrix convention is different for this PCA method. Normaly, each row corresponds to a single observation. HERE, each COLUMN corresponds to a diffrent observation
    @printf "PCA dimension reduction from %d to %d \n" size(digitPCA)[1] size(digitPCA)[2]

    Xtrain_reduced = MultivariateStats.predict(digitPCA, Xtrain)

    # need to rescale PCA features to be within generator's output domain
    minPCvals = zeros(numPCs)
    maxPCvals = zeros(numPCs)

    for i = 1:numPCs
        minPCvals[i] = minimum(Xtrain_reduced[i,:])
        maxPCvals[i] = maximum(Xtrain_reduced[i,:])
    end



    # validation and testing data
    if dataset == "MNIST"
        postTrainData = MNIST(; Tx=Float32, split=:test, dir="MNIST")
    elseif dataset == "FashionMNIST"
        postTrainData = FashionMNIST(; Tx=Float32, split=:test, dir="FashionMNIST")
    end

    postTrainData = postTrainData[postTrainData.targets .== digitClass]
    numObservations = size(postTrainData.features)[3]

    # postTrainData = shuffle(postTrainData)
    valData = postTrainData.features[:,:,1:div(numObservations, 2)]
    testData = postTrainData.features[:,:,(div(numObservations, 2)+1):end]

    Xval = reshape(valData, (28*28, div(numObservations, 2) ))
    Xtest = reshape(testData, (28*28, numObservations - div(numObservations, 2)))


    #load data files
    optParams = []
    optPositions = []
    optH = []
    detune_pulseList = [expDecayPulse, expPulse, gaussianPulse, linearPulse, reverseLinearPulse, sinPulse, trapezoidPulse, trianglePulse]
    rabi_pulseList = [gaussianPulse, trapezoidPulse, trianglePulse]


    trialNames = readdir(dirName, sort=false)
    trialNames = trialNames[contains.(trialNames, ".jld2")]
    trialPulses = []

    for rabiPulse in rabi_pulseList
        for detunePulse in detune_pulseList

            fileName = string(rabiPulse) * string(detunePulse) * "0.jld2"

            if any(fileName .== trialNames)
                
                println("loading " * fileName)
                if variant == "Global"
                    @load  dirName*"/"*fileName foundParams foundPositions
                else
                    @load  dirName*"/"*fileName foundParams foundPositions foundH
                    push!(optH, foundH)
                end

                push!(trialPulses, [rabiPulse, detunePulse])

                push!(optParams, foundParams)
                push!(optPositions, foundPositions)
                

            end

        end
    end

    ## inference step using collected training training_results

    # ideal inference
    if dataIsSaved
        @load "generatedPCs/" * dataset*"_"*variant *"/" *dataset*"_"*"ideal"*"_"* string(digitClass)* ".jld2" generatedPCs
    else
        generatedPCs= genIndivImages(optParams, optPositions, trialPulses, digitPCA, maxPCvals, minPCvals; optH=optH, noiseFlag=:ideal)
    
        JLD2.jldsave("generatedPCs/" * dataset*"_"*variant *"/" *dataset*"_"*"ideal"*"_"* string(digitClass)* ".jld2", generatedPCs=generatedPCs)
    end

    # bruteForceEnsembleFormation(generatedPCs, Xval, trialPulses, digitPCA)

    chosenIndices = greedyEnsembleFormation(generatedPCs, Xval, trialPulses, digitPCA)

    ideal_learners = trialPulses[chosenIndices]

    println(ideal_learners)

    ensemblePCS = mean(generatedPCs[chosenIndices, :, :], dims=1)[1,:,:]

    ensembleImages = genImagesFromPC(ensemblePCS, digitPCA; plotFlag=true, plotTitle="Ideal Simulation")

    ideal_testing_FID = fid_score(ensembleImages, Xtest)

    println(trialPulses[chosenIndices, :])

    vars = make_pdf(ensembleImages)

    avg_ideal_var = mean(vars)

    histogram(vars', normalize=:pdf, label="prob(Var)") # bins=range(0, 70, 20))
    xlabel!("Var")

    cumulative, prob = make_cdf(ensembleImages)

    cdf_plot = Plots.plot!(cumulative', prob, label="CDF")
    display(cdf_plot)

    flattened_images = copy(ensembleImages)

    flattened_images[flattened_images .>= 0.5] .= 1.0
    flattened_images[flattened_images .< 0.5] .= 0.0

    flattened_imagesPlots = Plots.Plot{}[]
            
    for k = plotRange
        h = heatmap(reshape(flattened_images[:, k], 28, 28)', yflip=true, color=:grays, aspect_ratio=1, axis=([], false), legend = :none)
        # h = imshow(reshape(outImages[:, i], 28, 28))
        push!(flattened_imagesPlots, h)
        # display(imagePlots[i])
    end

    flattened_digits_plot = Plots.plot(flattened_imagesPlots... )

    # display(flattened_digits_plot)

    # noisy inference
    if dataIsSaved
        @load "generatedPCs/" * dataset*"_"*variant *"/" *dataset*"_"*"noisy"*"_"* string(digitClass)* ".jld2" generatedPCs
    else
        generatedPCs= genIndivImages(optParams, optPositions, trialPulses, digitPCA, maxPCvals, minPCvals; optH=optH, noiseFlag=:noisy)

        JLD2.jldsave("generatedPCs/" * dataset*"_"*variant *"/" *dataset*"_"*"noisy"*"_"* string(digitClass)* ".jld2", generatedPCs=generatedPCs)
    end

    chosenIndices = greedyEnsembleFormation(generatedPCs, Xval, trialPulses, digitPCA)

    noisy_learners = trialPulses[chosenIndices]

    println(noisy_learners)

    ensemblePCS = mean(generatedPCs[chosenIndices, :, :], dims=1)[1,:,:]

    ensembleImages = genImagesFromPC(ensemblePCS, digitPCA; plotFlag=true, plotTitle="Error Prone Simulation")

    noisy_testing_FID = fid_score(ensembleImages, Xtest)

    println(trialPulses[chosenIndices, :])

    vars = make_pdf(ensembleImages)

    avg_noisy_var = mean(vars)

    histogram(vars', normalize=:pdf, label="prob(Var)") # bins=range(0, 70, 20))
    xlabel!("Var")

    cumulative, prob = make_cdf(ensembleImages)

    # cdf_plot = Plots.plot!(cumulative', prob, label="CDF")
    # display(cdf_plot)

    flattened_images = copy(ensembleImages)

    flattened_images[flattened_images .>= 0.5] .= 1.0
    flattened_images[flattened_images .< 0.5] .= 0.0

    flattened_imagesPlots = Plots.Plot{}[]
            
    for k =plotRange
        h = heatmap(reshape(flattened_images[:, k], 28, 28)', yflip=true, color=:grays, aspect_ratio=1, axis=([], false), legend = :none)
        # h = imshow(reshape(outImages[:, i], 28, 28))
        push!(flattened_imagesPlots, h)
        # display(imagePlots[i])
    end

    flattened_digits_plot = Plots.plot(flattened_imagesPlots... )

    # display(flattened_digits_plot)

    return ideal_testing_FID, noisy_testing_FID, ideal_learners, noisy_learners, avg_ideal_var, avg_noisy_var
end


ReCon_ideal_FIDs = []
ReCon_noisy_FIDs = []
ReCon_ideal_vars = []
ReCon_noisy_vars = []

ReCon_ideal_pulseSelection = []
ReCon_noisy_pulseSelection = []

for class in classRange
    ideal_fid, noisy_fid, ideal_learners, noisy_learners, ideal_var, noisy_var = single_ensemble_results(class)

    println("Ideal fid:", ideal_fid, "Noisy fid:", noisy_fid)

    println("Ideal Learners", string(ideal_learners))
    println("noisy learners", string(noisy_learners))

    println("Ideal var:", ideal_var)
    println("Noisy var:", noisy_var)
    push!(ReCon_ideal_FIDs, ideal_fid)
    push!(ReCon_noisy_FIDs, noisy_fid)
    push!(ReCon_ideal_vars, ideal_var)
    push!(ReCon_noisy_vars, noisy_var)

end

concatFIDs = [ReCon_ideal_FIDs  ReCon_noisy_FIDs]

# CSV.write("FIDscores/"*dataset*"_"*variant*"FIDscores.csv", Tables.table(concatFIDs), header=["Ideal", "Noisy"])

concatVars = [ReCon_ideal_vars  ReCon_noisy_vars]

CSV.write("varscores/"*dataset*"_"*variant*"varscores.csv", Tables.table(concatVars), header=["Ideal", "Noisy"])



