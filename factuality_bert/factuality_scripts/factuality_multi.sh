# train multitask model on all the datasets
seed=$1
set -e
OVERRIDES="exp_name = EXP_multi_task_factuality"
OVERRIDES+=", run_name = \"factbank,meantime,uw,uds-ih2,CB,rp,mv2--${seed}\""
OVERRIDES+=", pretrain_tasks = \"factbank,meantime,uw,uds-ih2,CB,rp,mv2\""
OVERRIDES+=", target_tasks = \"factbank,meantime,uw,uds-ih2,CB,rp,mv2\""
OVERRIDES+=", do_pretrain = 1"
OVERRIDES+=", random_seed=$seed"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 0"
OVERRIDES+=", val_interval = 2000"
OVERRIDES+=", cuda = auto"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune"
OVERRIDES+=", lr = .00001, min_lr = .0000001, lr_patience = 4, dropout=0.1, patience=5, max_epochs = 20"
OVERRIDES+=", input_module=bert-large-cased"


/home/jmurzaku/miniconda3/envs/jiant/bin/python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", do_full_eval = 1"

/home/jmurzaku/miniconda3/envs/jiant/bin/python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"



