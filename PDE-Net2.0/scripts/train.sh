#!/bin/bash
cd ../
python pretrain.py --config_file_path ./config.yaml --device_target ${DEVICE} --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs && \
python train.py --config_file_path ./config.yaml --device_target ${DEVICE} --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs
