models=('resnet20' 'sepreresnet20' 'densenet40_k12')

# FGSM
for m in ${models[@]}; do
    echo "------ $m - FGSM ------"
    python3 src/main.py attack \
        --source_dir data/cifar-10_eval \
        --output_dir experiments/attack/fgsm-${m} \
        --proxy_model $m \
        --target_method untargeted \
        --num_iters 1
done

# I-FGSM
for m in ${models[@]}; do
    echo "------ $m - IFGSM ------"
    python3 src/main.py attack \
        --source_dir data/cifar-10_eval \
        --output_dir experiments/attack/ifgsm-${m} \
        --proxy_model $m \
        --target_method untargeted \
        --num_iters 4 \
        --step_size divide
done

# MI-FGSM
for m in ${models[@]}; do
    echo "------ $m - MIFGSM ------"
    python3 src/main.py attack \
        --source_dir data/cifar-10_eval \
        --output_dir experiments/attack/mifgsm-${m} \
        --proxy_model $m \
        --target_method untargeted \
        --num_iters 4 \
        --decay_factor auto \
        --step_size divide
done

# PGD
for m in ${models[@]}; do
    echo "------ $m - PGD ------"
    python3 src/main.py attack \
        --source_dir data/cifar-10_eval \
        --output_dir experiments/attack/pgd-${m} \
        --proxy_model $m \
        --target_method untargeted \
        --num_iters 4
done
