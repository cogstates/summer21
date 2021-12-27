#!/bin/bash
# Run single task models on a dataset
# Usage: ./factuality_scripts/factuality_single_task.sh rp 42
# To run the Shared model, do: ./factuality_scripts/factuality_single.sh all-factuality

set -e
TASK=$1
seed=$2

OVERRIDES="exp_name = EXP_single_task_factuality"
OVERRIDES+=", run_name = single-${TASK}--${seed}"
OVERRIDES+=", pretrain_tasks = ${TASK}"
OVERRIDES+=", target_tasks = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", random_seed = $seed"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", cuda = auto"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
OVERRIDES+=", lr = .00001, min_lr = .0000001, dropout=0.1, max_epochs = 30"
OVERRIDES+=", reload_tasks=1, reload_indexing=1, reload_vocab=1, reindex_tasks=${TASK}"
OVERRIDES+=", input_module=bert-base-cased"

/home/jmurzaku/miniconda3/envs/jiant/bin/python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", delete_checkpoints_when_done = 0"

/home/jmurzaku/miniconda3/envs/jiant/bin/python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
