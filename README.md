# ReCon

Code accompanying the paper "ReCon: Reconfiguring Analog Rydberg Atom Quantum Computers for Quantum Generative Adversarial Networks" https://doi.org/10.1145/3676536.3676697

The implementation code is written in the Julia language (version 1.10.2).

## Training
You can run the code to train a GAN for a single class of a data set by running the following in your terminal:

```
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET $CLASSNUM $RABIPULSE $DETUNEPULSE
```

where `$DATASET` is the dataset to be used (normally "MNIST" or "FashionMNIST"), `$CLASSNUM` is an integer [0, 9] representing the digit/clothing class, `$RABIPULSE` is the pulse function/shape to be used for the Rabi frequency, and `$DETUNE` is the pulse function/shape to be used for the detuning. The available pulse functions are specified in pulseLibrary.jl. Note that there are hardware constraints on which pulses can be used which are explained in the paper.

For convenience, you can run the shell script train.sh, which will execute the above for all classes in a given dataset.

The first time you run this, you may be prompted to download the MNIST or FashionMNIST datasets.

## Plotting Images / Evaluation
You can run the code to plot images and compute evaluation metrics by running the file `ensemblePlotting.jl` . To save time, this script uses previously computed and saved PCA feature weights, so that it does not need to rerun the Hamiltonian simulation each time. These weights are saved in the folder `generatedPCs`. If you would like to regenerate the PCA features instead, you can do so by setting `dataIsSaved = false`.

## Real Hardware Runs
Real hardware runs are done on the Aquila computer from QuEra, which we accessed using the python interface for Amazon Braket. The code used to run this is in `ReConAquila.ipynb`. The parameters used for each run are saved in the folder `pythonInput`. The python requirements for this notebook can be found in `python_requirements.txt`

## Requirements
Simulation and evaluation code written in Julia v1.10.2. All package data is stored within the juliaNeutralAtom folder.

Code for interfacing with Aquila via Braket written in python 3.12.1. Package requirements are stored in python_requirements.txt

## Copyright
Copyright © 2024 Positive Technology Lab. All rights reserved. For permissions, contact ptl@rice.edu.
