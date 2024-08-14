# ReCon

Generative Adversarial Networks on Analog Rydberg atom computers.

All implementation code is written in the Julia language (version 1.10.2).

Code can be executed in the Julia REPL, after activating the environment.
To activate the environment: 

1. in the Julia REPL, type "[" and enter to launch the package manager. 
2. Run "activate juliaNeutralAtom"
3. hit delete to return to the Julia repl

From here, you can run the code to train a GAN for a single class of a data set by running:

rydbergEnsemble.jl $DATASET $CLASSNUM $RABIPULSE $DETUNEPULSE

where $DATASET is the dataset to be used (normally "MNIST" or "FashionMNIST"), $CLASSNUM is an integer [0, 9] representing the digit/clothing class, $RABIPULSE is the pulse function/shape to be used for the Rabi frequency, and $DETUNE is the pulse function/shape to be used for the detuning. The available pulse functions are specified in pulseLibrary.jl. Note that there are hardware constraints on which pulses can be used which are explained in the paper.

For convenience, you can run the shell script train.sh, which will execute the above for all classes.