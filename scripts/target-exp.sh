models=('resnet20' 'resnext29_32x4d' 'densenet40_k12')
targets=('untargeted' 'random' 'next')

for m in ${models[@]}; do
    for t in ${targets[@]}; do
        echo "------ $m - $t ------"
        python3 src/main.py attack \
            --source_dir data/cifar-10_eval \
            --proxy_model $m \
            --target_method $t \
            --num_iters 1
    done
done
