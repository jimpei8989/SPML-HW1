# Security and Privacy of Machine Learning - Homework 1
> Gray-box attacks on CIFAR-10 datasets

## Environment
- Python version: `3.8.5`
- Framework: `PyTorch 1.6.0`
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

### Help
```
usage: main.py [-h] [--source_dir SOURCE_DIR] [--output_dir OUTPUT_DIR] [--target_model TARGET_MODEL] [--epsilon EPSILON] [--num_iters NUM_ITERS] [--target_method TARGET_METHOD] task

positional arguments:
  task                  choose one from {attack, evaluate}

optional arguments:
  -h, --help            show this help message and exit
  --source_dir SOURCE_DIR
                        the directory of the original validating images
  --output_dir OUTPUT_DIR
                        the directory of the output adversarial images
  --target_model TARGET_MODEL
                        proxy model for generating adversarial images
  --epsilon EPSILON     the l-infinity value in [0, 1], default 8/256 = 0.03125
  --num_iters NUM_ITERS
                        number of iterations for iterative FGSM, default 1
  --target_method TARGET_METHOD
                        method for target generation, choose one from {negative, random, next}, default negative
```

### Attacking Single Model
```bash
python3 src/main.py attack --source_dir ${source_dir} --output_dir ${output_dir} --target_model ${target_model} --target_method ${target_model} --num_iters {}
```

### Evaluation
I use the following model for evalution my adversarial examples:
- resnet110_cifar10
- resnet272bn_cifar10
- preresnet272bn_cifar10
- resnext29_32x4d_cifar10
- seresnet272bn_cifar10
- pyramidnet110_a48_cifar10
- densenet40_k36_bc_cifar10
- wrn16_10_cifar10
- ror3_164_cifar10
- shakeshakeresnet26_2x32d_cifar10

They are chosen mainly based on #params and GFLOPs, since I don't have powerful computational resource QQ.
