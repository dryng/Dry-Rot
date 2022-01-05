#!/bin/bash
models_w_dataloader="custom_mobilenet_v3_small efficient_net_b4 mobilenet_v3_large mobilenet_v3_small"
models_wo_dataloader="custom_mobilenet_v3_small_untrained efficient_net_b3"

for val in $models_w_dataloader; do
    python train.py $val >> ../metrics/precision_recall.txt
done
