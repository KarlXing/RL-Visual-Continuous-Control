#!/bin/bash

(python src/train.py  --agent sac    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 1) &
(python src/train.py  --agent sac    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 2) &
(python src/train.py  --agent sac    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 3) &
wait;

(python src/train.py  --agent sacae  --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 1) &
(python src/train.py  --agent sacae  --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 2) &
(python src/train.py  --agent sacae  --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 3) &
wait;

(python src/train.py  --agent curl   --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 1) &
(python src/train.py  --agent curl   --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 2) &
(python src/train.py  --agent curl   --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 3) &
wait;

(python src/train.py  --agent rad    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 1) &
(python src/train.py  --agent rad    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 2) &
(python src/train.py  --agent rad    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 3) &
wait;

(python src/train.py  --agent drq    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 1) &
(python src/train.py  --agent drq    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 2) &
(python src/train.py  --agent drq    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 3) &
wait;

(python src/train.py  --agent atc    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 1) &
(python src/train.py  --agent atc    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 2) &
(python src/train.py  --agent atc    --domain_name cartpole  --task_name swingup  --num_train_steps 25000   --action_repeat 8  --batch_size  512   --seed 3) &
wait;