# Security and Privacy of Machine Learning - Homework 1
> Gray-box attacks on CIFAR-10 datasets

## Environment
- Python version: `3.8.5`
- DL Framework: `PyTorch 1.6.0`
- Packages: Please refer to [requirements.txt](./requirements.txt)

### Building Up the Environment
```bash
# Assume you already have pyenv :)
$ pyenv install 3.8.5
$ pyenv virtualenv 3.8.5 SPML-HW1
$ pyenv local SPML-HW1
$ pip3 install -r requirements.txt
```

## How to run my script?

### Prerequisite
- Download the dataset into `data/cifar-10_eval`

### Help Message
Please take a look at:
```sh
python3 src/main.py --help
```

### Attacking Single Model
```sh
proxy_model="resnet20"

python3 src/main.py attack \
    --source_dir data/cifar-10_eval \
    --output_dir adv_imgs \
    --proxy_model ${proxy_model} \
    --target_method untargeted \
    --num_iters 8 \
    --eval_set small
```

### Evaluation with defense (JPEG-80)
```sh
python3 src/main.py evaluate \
    --source_dir data/cifar-10_eval \
    --output_dir adv_imgs \
    --eval_set large \
    --defense JPEG-80
```
- If you don't want to defense, simply remove that argument.

### Generating my adversarial examples
```sh
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
```

> Note that some log / experiment result files will appear in `adv_imgs/` directory. (deleted)

## Final Evaluation
The models used in this experiment are:
- nin
- sepreresnet56
- xdensenet40-2-k24-bc
- ror3-110
- resnet1001

### Vanilla Evaluation (w/o preprocessing)

| Models     | nin | sepreresnet56 | resnet1001 | xdensenet40-2-k24-bc | ror3-110 |
| ---------- |------|------|------|------|------|
| airplane   | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| automobile | 0.40 | 0.00 | 0.00 | 0.10 | 0.10 |
| bird       | 0.10 | 0.10 | 0.00 | 0.00 | 0.10 |
| cat        | 0.00 | 0.00 | 0.00 | 0.10 | 0.10 |
| deer       | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| dog        | 0.20 | 0.10 | 0.00 | 0.00 | 0.20 |
| frog       | 0.20 | 0.00 | 0.00 | 0.00 | 0.10 |
| horse      | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| ship       | 0.60 | 0.10 | 0.00 | 0.10 | 0.00 |
| truck      | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| *Mean*     | 0.27 | 0.03 | 0.00 | 0.03 | 0.06 | 

Overall: mean = 0.0780, std = 0.0979

