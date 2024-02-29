# Pareto Auxiliary-Task Learning
# tasks: c i p q r s

# wandb group name for each experiment: "domainnet-{args.arch}-{args.target[0]}"
architecture=HPS
dataset=nyuv2

# 1. local training
# a) load model from checkpoint `{log_dir}/checkpoints/round={round_idx-1}_merged.pth`
#    if round_idx=0, then load the pre-trained model from torchvision.
# b) train the K+1 models for each task and save the models to `{log_dir}/checkpoints/round={round_idx}_task={task_name}_trained.pth`
# Usage: pareto_atl_train <round_idx> <task_name>
function pareto_atl_train {
    round_idx=$1
    task_name=$2

    python pareto_atl_train.py \
        --arch ${architecture} --scheduler step \
        --epoch_step 30 --seed 0 \
        --source_tasks segmentation depth normal --target_tasks ${target_task} \
        --round_idx ${round_idx} --task_name ${task_name} \
        --log_dir logs/${dataset}/${architecture}/${target_task} \
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

    save_interval=5
    python pareto_atl_merge.py \
        --arch ${architecture} --scheduler step \
        --epoch_step 10 --seed 0 --lr 1e-3 --weight_decay 0\
        --source_tasks segmentation depth normal --target_tasks ${target_task} \
        --round_idx ${round_idx} --task_name ${target_task} \
        --log_dir logs/${dataset}/${architecture}/${target_task} \
        --run_name pareto_atl_merge-${target_task}-${round_idx}
}

# Evaluate the test accuracy of the model load from the checkpoint on the target_task,
# and save the result will be saved in the ckpt_path directory as csv file.
# Usage: pareto_atl_test <ckpt_path>
function pareto_atl_test {
    ckpt_path=$1
    python pareto_atl_test.py \
        --arch ${architecture} --scheduler step \
        --source_tasks segmentation depth normal --target_tasks ${target_task} \
        --ckpt ${ckpt_path}
}

function pareto_atl {
    for round_idx in {0..5}
    do 
        # train K+1 models for each task
        CUDA_VISIBLE_DEVICES=1 pareto_atl_train ${round_idx} segmentation &
        CUDA_VISIBLE_DEVICES=2 pareto_atl_train ${round_idx} depth &
        CUDA_VISIBLE_DEVICES=3 pareto_atl_train ${round_idx} normal &
        wait
        # merge the models
        CUDA_VISIBLE_DEVICES=7 pareto_atl_merge ${round_idx}
        # test the merged model
        for epoch_idx in $(seq 0 5 10)
        do
            CUDA_VISIBLE_DEVICES=7 pareto_atl_test logs/${dataset}/${architecture}/${target_task}/checkpoints/round=${round_idx}_epoch=${epoch_idx}_merged.pth
        done
        CUDA_VISIBLE_DEVICES=7 pareto_atl_test logs/${dataset}/${architecture}/${target_task}/checkpoints/round=${round_idx}_merged.pth
        wait
    done
}


for target_task in segmentation depth normal
do
    pareto_atl
done
