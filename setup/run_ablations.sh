#!/bin/bash
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
L=logs/stdout; mkdir -p "$L"
B=(-m src.train --model own --lr 3e-3 --epochs 15 --patience 15 --phase ablation)
run() { name=$1; shift; "$P" "${B[@]}" "$@" --run-name "$name" > "$L/$name.log" 2>&1; echo "$name exit $?"; }
run own_e15_s656 --seed 656
run own_e15_s657 --seed 657
run own_abl_pool_avg --pool avg
run own_abl_no_c5 --no-c5
run own_abl_48x64 --size 48x64
run own_abl_width_half --widths 8,16,32
run own_abl_amp --amp
