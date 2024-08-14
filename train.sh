#!/bin/zsh
DATASET=$1
RABIPULSE=$2
DETUNEPULSE=$3

julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 0 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 1 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 2 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 3 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 4 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 5 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 6 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 7 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 8 $RABIPULSE $DETUNEPULSE
julia --project=./juliaNeutralAtom rydbergEnsemble.jl $DATASET 9 $RABIPULSE $DETUNEPULSE
