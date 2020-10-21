proxy_models=(
    'resnet20'
    'resnet1001'
    'sepreresnet20'
    'sepreresnet542bn'
    'densenet40_k12'
    'densenet100_k24'
    'pyramidnet110_a48'
    'resnext29_32x4d'
    'nin'
)

iters=(2 4 8 16 32)

for i in ${iters[@]}; do
    echo $i
    python3 src/main.py attack \
        --source_dir data/cifar-10_eval \
        --output_dir experiments/final/pgd-$i \
        --proxy_model ${proxy_models[@]} \
        --target_method untargeted \
        --num_iters $i \
        --eval_set large

    python3 src/main.py evaluate \
        --source_dir data/cifar-10_eval \
        --output_dir experiments/final/pgd-$i \
        --eval_set large \
        --defense JPEG-80
done
