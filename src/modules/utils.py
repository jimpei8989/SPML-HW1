from pytorchcv.model_provider import _models, get_model

all_pytorchcv_cifar10_models = [k for k in _models if k.endswith('cifar10')]

all_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

target_models = [
    'resnet110_cifar10',            # Top1Err: 3.69 / Params: 1.7M / FLOPs: 255M
    'preresnet272bn_cifar10',       # Top1Err: 3.25 / Params: 2.8M / FLOPs: 420M
    'pyramidnet110_a48_cifar10',    # Top1Err: 3.72 / Params: 1.7M / FLOPs: 408M
    'densenet40_k36_bc_cifar10',    # Top1Err: 4.04 / Params: 1.5M / FLOPs: 654M
]

evaluation_models = [

]
