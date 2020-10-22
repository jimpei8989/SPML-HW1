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

iters=32

python3 src/main.py attack \
    --source_dir data/cifar-10_eval \
    --output_dir adv_imgs \
    --proxy_model ${proxy_models[@]} \
    --target_method untargeted \
    --num_iters ${iters} \
    --eval_set large

python3 src/main.py evaluate \
    --source_dir data/cifar-10_eval \
    --output_dir adv_imgs \
    --eval_set large \
    --defense JPEG-80

mv adv_imgs/result-large-defense.md adv_imgs/results-large-d80.md

python3 src/main.py evaluate \
    --source_dir data/cifar-10_eval \
    --output_dir adv_imgs \
    --eval_set large \
    --defense JPEG-60

mv adv_imgs/result-large-defense.md adv_imgs/results-large-d60.md
