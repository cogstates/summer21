#!/bin/bash
# Train multi-task models on random splits of CB, rp, mv2
# Usage: for i in `seq 2 3`; do ./factuality_scripts/factuality_multi_crossval.sh $i; done

set -e
split=${1:-"2"}
seed=${2:-"0"}

OVERRIDES="exp_name = EXP_multi_task_factuality"
OVERRIDES+=", run_name = \"cb-rp-mv2_multi___split${split}--${seed}\""
OVERRIDES+=", pretrain_tasks = \"factbank,meantime,uw,uds-ih2,CB___split${split},rp___split${split},mv2___split${split}\""
OVERRIDES+=", target_tasks = \"CB___split${split},rp___split${split},mv2___split${split}\""
OVERRIDES+=", cuda = auto"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune"
OVERRIDES+=", lr = .00001, min_lr = .0000001, lr_patience = 4, dropout=0.1, patience=5, max_epochs = 20"
OVERRIDES+=", input_module=bert-large-cased"
OVERRIDES+=", do_pretrain = 1"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", random_seed=${seed}"
OVERRIDES+=", do_full_eval = 0"

pushd "${PWD%jiant*}jiant"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", target_tasks = \"CB___split${split},rp___split${split},mv2___split${split}\""
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", delete_checkpoints_when_done = 0"
python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"


