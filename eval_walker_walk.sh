#!/bin/bash

(python src/train.py  --agent sac    --domain_name walker  --task_name walk  --num_train_steps 500000   --action_repeat 2  --batch_size  512   --seed 1) &
wait;

(python src/train.py  --agent sacae  --domain_name walker  --task_name walk  --num_train_steps 500000   --action_repeat 2  --batch_size  512   --seed 1) &
wait;

(python src/train.py  --agent curl   --domain_name walker  --task_name walk  --num_train_steps 500000   --action_repeat 2  --batch_size  512   --seed 1) &
wait;

(python src/train.py  --agent rad    --domain_name walker  --task_name walk  --num_train_steps 500000   --action_repeat 2  --batch_size  512   --seed 1) &
wait;

(python src/train.py  --agent drq    --domain_name walker  --task_name walk  --num_train_steps 500000   --action_repeat 2  --batch_size  512   --seed 1) &
wait;

(python src/train.py  --agent atc    --domain_name walker  --task_name walk  --num_train_steps 500000   --action_repeat 2  --batch_size  512   --seed 1) &
wait;