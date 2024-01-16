#!/usr/bin/env bash

CONFIG=$1
CARDS=$2  # 代表训练使用到的卡数

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mpirun \
    --allow-run-as-root \
    -n $CARDS \
    python tools/train.py \
    --config $CONFIG
    ${@:3}
