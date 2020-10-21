from pytorchcv.model_provider import _models

all_pytorchcv_cifar10_models = [k for k in _models if k.endswith('cifar10')]

all_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

proxy_models = list(map(lambda s: s + '_cifar10', [
    'nin',
    'resnet20',
    'preresnet20',
    'resnext29_32x4d',
    'seresnet20',
    'densenet40_k12',
    'ror3_56',
    'shakeshakeresnet20_2x16d',
]))


eval_models = {
    'small': list(map(lambda s: s + '_cifar10', [
        'resnet20',
        'sepreresnet20',
        'densenet40_k12',
        'nin',
        'resnext29_32x4d',
        'pyramidnet110_a48',
    ])),
    'proxy_exp': list(map(lambda s: s + '_cifar10', [
        'resnet20',
        'resnet1001',
        'sepreresnet20',
        'sepreresnet542bn',
        'densenet40_k12',
        'densenet100_k24',
        'pyramidnet110_a48',
    ])),
    'large': list(map(lambda s: s + '_cifar10', [
        'nin',
        'resnet20',
        'resnet1001',
        'resnet164bn',
        'preresnet20',
        'resnext29_32x4d',
        'seresnet20',
        'pyramidnet110_a48',
        'densenet40_k12',
        'xdensenet40_2_k24_bc',
        'ror3_56',
        'shakeshakeresnet20_2x16d',
        'diaresnet20',
    ]))
}

# proxy_models = [
#     'resnet110_cifar10',            # Top1Err: 3.69 / Params: 1.7M / FLOPs: 255M
#     'preresnet272bn_cifar10',       # Top1Err: 3.25 / Params: 2.8M / FLOPs: 420M
#     'resnext29_32x4d_cifar10',      # Top1Err: 3.15 / Params: 4.7M / FLOPs: 780M
#     'pyramidnet110_a48_cifar10',    # Top1Err: 3.72 / Params: 1.7M / FLOPs: 408M
#     'densenet40_k36_bc_cifar10',    # Top1Err: 4.04 / Params: 1.5M / FLOPs: 654M
# ]

# eval_models = [
#     'resnet110_cifar10',            # Top1Err: 3.69 / Params: 1.7M / FLOPs: 255M
#     'resnet272bn_cifar10',
#     'preresnet272bn_cifar10',       # Top1Err: 3.25 / Params: 2.8M / FLOPs: 420M
#     'resnext29_32x4d_cifar10',      # Top1Err: 3.15 / Params: 4.7M / FLOPs: 780M
#     'seresnet272bn_cifar10',
#     'pyramidnet110_a48_cifar10',    # Top1Err: 3.72 / Params: 1.7M / FLOPs: 408M
#     'densenet40_k36_bc_cifar10',    # Top1Err: 4.04 / Params: 1.5M / FLOPs: 654M
#     'wrn16_10_cifar10',
#     'ror3_164_cifar10',
#     'shakeshakeresnet26_2x32d_cifar10',
# ]
