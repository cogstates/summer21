#!/bin/bash
# Run a single task model with MNLI transfer (singleM)
# Usage: ./factuality_singleM.sh CB 42 MNLI_CHECKPOINT_PATH
# To run the SharedM model, do: ./factuality_scripts/factuality_singleM.sh all-factuality MNLI_CHECKPOINT_PATH

set -e
TASK=${1:-"CB"}
seed=$2
MNLI_CHECKPOINT=${3:-"bert-mnli/tuning-0/model_state_pretrain_val_76.best.th"}

OVERRIDES="exp_name = EXP_mnli_transfer_factuality"
OVERRIDES+=", run_name = ${TASK}--${seed}"
OVERRIDES+=", pretrain_tasks = ${TASK}"
OVERRIDES+=", target_tasks = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", cuda = 0"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
OVERRIDES+=", lr = .00001, min_lr = .0000001, dropout=0.1, max_epochs = 20"
OVERRIDES+=", input_module=bert-large-cased"
OVERRIDES+=", reload_tasks=1, reload_indexing=1, reload_vocab=1, reindex_tasks=${TASK}"
## LOAD MNLI CHECKPOIONT
OVERRIDES+=", load_target_train_checkpoint = ${MNLI_CHECKPOINT}"
OVERRIDES+=", random_seed = $seed"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", target_tasks = \"factbank,meantime,uw,uds-ih2,CB,rp,mv2_2200\""
OVERRIDES+=", use_classifier = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", delete_checkpoints_when_done = 1"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
