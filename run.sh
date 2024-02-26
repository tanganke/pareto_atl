# pareto adamerging
python scripts/clip_pareto_adamerging.py pareto_tta=false evaluate=true \
    checkpoint_dir="/data0/users/tanganke/projects/pareto_merging/outputs/clip_pareto_adamerging/2024-02-03_15-03-20/checkpoints" \
    num_workers=8 batch_size=64

python scripts/clip_pareto_adamerging.py pareto_tta=true evaluate=true \
    seen_datasets="[SUN397,Cars,RESISC45,DTD,SVHN,GTSRB]" test_datasets="[SUN397,Cars,RESISC45,DTD,SVHN,GTSRB,MNIST,EuroSAT]" \
    num_workers=8 batch_size=64 lr=1e-3

python scripts/clip_pareto_adamerging.py pareto_tta=true evaluate=true \
    seen_datasets="[SUN397,Cars,GTSRB,EuroSAT,DTD,MNIST]" test_datasets="[SUN397,Cars,GTSRB,EuroSAT,DTD,MNIST,RESISC45,SVHN]" \
    num_workers=8 batch_size=64 lr=1e-3