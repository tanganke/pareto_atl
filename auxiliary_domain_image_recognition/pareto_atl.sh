# Pareto Auxiliary-Task Learning
# tasks: c i p q r s

# wandb group name for each experiment: "domainnet-{args.arch}-{args.target[0]}"
train_iters_per_epoch=2500
merge_iters_per_epoch=200
merge_save_interval=50
# 1. local training
# a) load model from checkpoint `{log_dir}/checkpoints/round={round_idx-1}_merged.pth`
#    if round_idx=0, then load the pre-trained model from torchvision.
# b) train the K+1 models for each task and save the models to `{log_dir}/checkpoints/round={round_idx}_task={task_name}_trained.pth`
# Usage: pareto_atl_train <round_idx> <task_name>
function pareto_atl_train {
    round_idx=$1
    task_name=$2

    python pareto_atl_train.py data/domainnet -s c i p q r s  \
    --workers 8 \
    -t ${target_task} -a ${architecture} \
    --round_idx ${round_idx} --task_name ${task_name} \
    -i ${train_iters_per_epoch} --lr 1e-3 --wd 5e-4 --optimizer adamw \
    --log_dir logs/domainnet/${architecture}/${target_task} \
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

    python pareto_atl_merge.py data/domainnet -s c i p q r s  \
    --seed 0 --workers 8 \
    -t ${target_task}  -a ${architecture} --round_idx ${round_idx} \
    -i ${merge_iters_per_epoch} --save_interval ${merge_save_interval} --lr 1e-2 --wd 0\
    --log_dir logs/domainnet/${architecture}/${target_task} \
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

function pareto_atl_helper {
    # train K+1 models for each task
    CUDA_VISIBLE_DEVICES=2 pareto_atl_train ${round_idx} c &
    CUDA_VISIBLE_DEVICES=3 pareto_atl_train ${round_idx} i &
    CUDA_VISIBLE_DEVICES=4 pareto_atl_train ${round_idx} p &
    CUDA_VISIBLE_DEVICES=5 pareto_atl_train ${round_idx} q &
    CUDA_VISIBLE_DEVICES=6 pareto_atl_train ${round_idx} r &
    CUDA_VISIBLE_DEVICES=7 pareto_atl_train ${round_idx} s &
    wait
    # merge the models
    CUDA_VISIBLE_DEVICES=7 pareto_atl_merge ${round_idx}
    # test the merged model
    for step_idx in $(seq 0 ${merge_save_interval} ${merge_iters_per_epoch})
    do
        CUDA_VISIBLE_DEVICES=7 pareto_atl_test logs/domainnet/${architecture}/${target_task}/checkpoints/round=${round_idx}_step=${step_idx}_merged.pth
    done
    CUDA_VISIBLE_DEVICES=7 pareto_atl_test logs/domainnet/${architecture}/${target_task}/checkpoints/round=${round_idx}_merged.pth
    wait
}

function pareto_atl {
    for round_idx in {0..2} # 3 rounds, warmup training
    do 
        train_iters_per_epoch=2500
        merge_iters_per_epoch=200
        merge_save_interval=50
        pareto_atl_helper
    done

    for round_idx in {3..5}
    do 
        train_iters_per_epoch=200
        merge_iters_per_epoch=200
        merge_save_interval=50
        pareto_atl_helper
    done

}

architecture=resnet101

for target_task in c i p q r s
do
    pareto_atl
done


# %% MTL
function pareto_atl_mtl_test {
    ckpt_path=$1
    output_path=$2

    python pareto_atl_test.py data/domainnet -d DomainNet -s c i p q r s \
    -t ${test_task} -a ${architecture} \
    --ckpt_path ${ckpt_path} --output_path ${output_path}
}

target_task=c
round=0
architecture=resnet101
for test_task in c i p q r s
do
    pareto_atl_mtl_test \
        logs/domainnet/resnet101/mtl/${target_task}/round=${round}_merged.pth \
        logs/domainnet/resnet101/mtl/${target_task}/round=${round}_merged-${test_task}.csv
    for task in c i p q r s
    do  
        pareto_atl_mtl_test \
            logs/domainnet/resnet101/mtl/${target_task}/round=${round}_task=${task}.pth \
            logs/domainnet/resnet101/mtl/${target_task}/round=${round}_task=${task}-${test_task}.csv
    done
done
