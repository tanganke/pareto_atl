# Pareto Auxiliary-Task Learning
# tasks: c i p q r s

# 1. local training
# a) load model from checkpoint `{log_dir}/checkpoints/round={round_idx-1}_merged.pth`
#    if round_idx=0, then load the pre-trained model from torchvision.
# b) train the K+1 models for each task and save the models to `{log_dir}/checkpoints/round={round_idx}_task={task_name}_trained.pth`
# Usage: pareto_atl_train <round_idx> <task_name>
function pareto_atl_train {
    round_idx=$1
    task_name=$2

    iters_per_epoch=2500
    python pareto_atl_train.py data/domainnet -s c i p q r s  \
    --seed 0 --workers 8 \
    -t ${target_task} --task_name ${task_name} -a ${architecture} \
    -i ${iters_per_epoch} --lr 0.01 \
    --log_dir logs/domainnet/pareto_atl/${target_task} \
    --run_name pareto_atl_train-${target_task}-${round_idx}-${task_name}
}

# 2. Pareto ATL merge
# a) load the model at round $round_idx, step 0 (`{log_dir}/checkpoints/round={round_idx-1}_merged.pth`) 
# and the K+1 models from checkpoint `{log_dir}/checkpoints/round={round_idx}_task={task_name}_trained.pth`.
# b) merge the models and save the merged model to `{log_dir}/checkpoints/round={round_idx}_merged.pth` 
# and `{log_dir}/checkpoints/round={round_idx}_step={step_idx+1}_merged.pth` every $save_interval steps.
# Usage: pareto_atl_merge <round_idx>
function pareto_atl_merge {
    round_idx=$1

    iters_per_epoch=20
    save_interval=5
    python pareto_atl_merge.py data/domainnet -s c i p q r s  \
    --seed 0 --workers 8 \
    -t ${target_task}  -a ${architecture} \
    -i ${iters_per_epoch} --save_interval ${save_interval} --lr 1e-3 \
    --log_dir logs/domainnet/pareto_atl/${target_task} \
    --run_name pareto_atl_merge-${target_task}-${round_idx}
}

# Evaluate the test accuracy of the model load from the checkpoint on the target_task,
# and save the result will be saved in the ckpt_path directory as csv file.
# Usage: pareto_atl_test <ckpt_path>
function pareto_atl_test {
    ckpt_path=$1
    python pareto_atl_test.py data/domainnet -d DomainNet -s c i p q r s \
    -t ${target_task} -a ${architecture} \
    --ckpt ${ckpt_path}
}

function pareto_atl {
    for round_idx in {0..19}
    do 
        # train K+1 models for each task
        CUDA_VISIBLE_DEVICES=0 pareto_atl_train ${round_idx} c &
        CUDA_VISIBLE_DEVICES=1 pareto_atl_train ${round_idx} i &
        CUDA_VISIBLE_DEVICES=2 pareto_atl_train ${round_idx} p &
        CUDA_VISIBLE_DEVICES=3 pareto_atl_train ${round_idx} q &
        CUDA_VISIBLE_DEVICES=4 pareto_atl_train ${round_idx} r &
        CUDA_VISIBLE_DEVICES=5 pareto_atl_train ${round_idx} s &
        wait
        # merge the models
        pareto_atl_merge ${round_idx}
        # test the merged model
        for step_idx in $(seq 0 5 20)
        do
            pareto_atl_test logs/domainnet/pareto_atl/${target_task}/checkpoints/round=${round_idx}_step=${step_idx}_merged.pth &
        done
        pareto_atl_test logs/domainnet/pareto_atl/${target_task}/checkpoints/round=${round_idx}_merged.pth &
        wait
    done
}