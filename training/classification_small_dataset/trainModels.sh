#!/bin/bash
gpuNum=$1
if [[ gpuNum -eq "0" ]]
then
    models="efficient_net_b3 efficient_net_b4"
else
    models="custom_mobilenet_v3_small mobilenet_v3_small mobilenet_v3_large"
fi
for val in $models; do
    echo "Training: ${val}"
    python train.py $gpuNum $val >> ${val}_log.txt
done