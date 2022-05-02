#!/bin/bash

for exp in 24 25
do
    echo "starting" $exp 
    date
    python segmentation_training.py ../experiments/$exp/model_config.json ../experiments/$exp/results_config.json
    date
    echo $exp "done"
done

for exp in 26 27 28 29 30 31
do
    echo "starting" $exp 
    date
    python classification_training.py ../experiments/$exp/model_config.json ../experiments/$exp/results_config.json
    date
    echo $exp "done"
done