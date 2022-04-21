#!/bin/bash

for exp in 11 12 13 14 15 16
do
    echo "starting" $exp 
    date
    python classification_training.py ../experiments/$exp/model_config.json ../experiments/$exp/results_config.json
    date
    echo $exp "done"
done