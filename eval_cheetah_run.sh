#!/bin/bash

(python src/train.py  --agent sac    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 1) &
(python src/train.py  --agent sac    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 2) &
(python src/train.py  --agent sac    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 3) &
wait;

(python src/train.py  --agent sacae  --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 1) &
(python src/train.py  --agent sacae  --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 2) &
(python src/train.py  --agent sacae  --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 3) &
wait;

(python src/train.py  --agent curl   --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 1) &
(python src/train.py  --agent curl   --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 2) &
(python src/train.py  --agent curl   --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 3) &
wait;

(python src/train.py  --agent rad    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 1) &
(python src/train.py  --agent rad    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 2) &
(python src/train.py  --agent rad    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 3) &
wait;

(python src/train.py  --agent drq    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 1) &
(python src/train.py  --agent drq    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 2) &
(python src/train.py  --agent drq    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 3) &
wait;

(python src/train.py  --agent atc    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 1) &
(python src/train.py  --agent atc    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 2) &
(python src/train.py  --agent atc    --domain_name cheetah  --task_name run  --num_train_steps 250000   --action-repeat 4  --seed 3) &
wait;