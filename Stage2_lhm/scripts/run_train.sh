#!/bin/bash

rm -rf ./exp/time.log

python -B ./train.py \
--tr_list=../examples/filelists/tr_list2.txt \
--cv_file=/data/lihaoming/gen_data/data/h5/tt/test2.ex \
--ckpt_dir=exp \
--time_log=./exp/time.log \
