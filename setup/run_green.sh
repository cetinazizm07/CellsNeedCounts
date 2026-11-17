#!/bin/bash
# Combined "green" configuration of the own CNN: half width + no c5 + fp16 AMP.
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
L=logs/stdout; mkdir -p "$L"
B=(-m src.train --model own --lr 3e-3 --epochs 15 --patience 15 --widths 8,16,32 --no-c5 --amp)
run() { name=$1; shift; "$P" "${B[@]}" "$@" --run-name "$name" > "$L/$name.log" 2>&1; echo "$name exit $?"; }
run own_green_96x128 --size 96x128
run own_green_48x64 --size 48x64
