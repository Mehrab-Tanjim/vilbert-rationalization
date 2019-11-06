#!/bin/bash

if [ $1 = 'debug' ];
then
    echo 'In Debug mode'
    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/pytorch_model_9.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 2e-5 --num_workers 2 --tasks 1-2 --save_name chk --debug

else
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/pytorch_model_9.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 2e-5 --num_workers 2 --tasks 1-2 --save_name pretrained 

fi
