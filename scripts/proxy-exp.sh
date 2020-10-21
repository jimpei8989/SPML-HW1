models=('resnet20' 'sepreresnet20' 'densenet40_k12' 'resnet1001' 'sepreresnet542bn' 'densenet100_k24')

# PGD
for m in ${models[@]}; do
    echo "------ $m - PGD ------"
    python3 src/main.py attack \
        --source_dir data/cifar-10_eval \
        --output_dir experiments/proxy/${m} \
        --proxy_model $m \
        --target_method untargeted \
        --num_iters 8 \
        --eval_set proxy_exp
done

echo "------ Ensemble - Weak - PGD ------"
python3 src/main.py attack \
    --source_dir data/cifar-10_eval \
    --output_dir experiments/proxy/ensemble-weak \
    --proxy_model resnet20 sepreresnet20 densenet40_k12 \
    --target_method untargeted \
    --num_iters 8 \
    --eval_set proxy_exp

echo "------ Ensemble - Strong - PGD ------"
python3 src/main.py attack \
    --source_dir data/cifar-10_eval \
    --output_dir experiments/proxy/ensemble-strong \
    --proxy_model resnet1001 sepreresnet542 densenet100_k24 \
    --target_method untargeted \
    --num_iters 8 \
    --eval_set proxy_exp

echo "------ Ensemble - All - PGD ------"
python3 src/main.py attack \
    --source_dir data/cifar-10_eval \
    --output_dir experiments/proxy/ensemble \
    --proxy_model ${models[@]} \
    --target_method untargeted \
    --num_iters 8 \
    --eval_set proxy_exp