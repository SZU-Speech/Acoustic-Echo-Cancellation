#!/bin/bash


test_step=true
eval_step=false


if $test_step; then
    python -B ./test.py \
        --gpu_ids=0 \
        --tt_list=../examples/filelists/tt_list.txt  \
        --ckpt_dir=exp \
        --model_file=./exp/models/best.pt
fi

if $eval_step; then
    # python -B ./measure.py --metric=stoi --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
    python -B ./measure.py --metric=pesq --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
    # -B ./measure.py --metric=snr --tt_list=../filelists/tt_list.txt --ckpt_dir=$ckpt_dir
fi
