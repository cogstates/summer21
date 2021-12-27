#!/bin/bash
# Train single models on random splits of a dataset
# Usage: for i in `seq 2 3`; do ./factuality_scripts/factuality_single_crossval.sh CB $i; done

set -e
task=${1:-"CB"}
split=${2:-"0"}
TASK="${task}___split${split}"
seed=${3:-"0"}

OVERRIDES="exp_name = EXP_single_task_factuality"
OVERRIDES+=", run_name = ${TASK}--${seed}"
OVERRIDES+=", pretrain_tasks = ${TASK}"
OVERRIDES+=", target_tasks = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", val_interval = 60" 
OVERRIDES+=", cuda = 0"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
OVERRIDES+=", lr = .00001, min_lr = .0000001, lr_patience = 4, dropout=0.1, max_epochs = 20"
OVERRIDES+=", input_module=bert-large-cased"
OVERRIDES+=", reload_tasks=1"
OVERRIDES+=", random_seed=${seed}, delete_checkpoints_when_done=1"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
