from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model

from modules.dataset import OriginalDataset
from modules.ensemble import Ensemble
from modules.fgsm import fgsm_attack
from modules.evaluate import check_adv_validity, evaluate_single_model
from modules.utils import all_labels, eval_models, proxy_models

SEED = 0x06902029


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def attack_root(proxy_models, source_dir, output_dir, **kwargs):
    model = Ensemble(proxy_models)

    ori_dataset = OriginalDataset(data_dir=source_dir)
    ori_dataloader = DataLoader(ori_dataset, batch_size=1)
    adv_dataset = fgsm_attack(model, ori_dataloader, output_dir, **kwargs)
    adv_dataset.save_to_directory()


def evaluate_root(source_dir, output_dir, epsilon, **kwargs):
    ori_dataset = OriginalDataset(data_dir=source_dir)
    adv_dataset = OriginalDataset(data_dir=output_dir)
    adv_dataloader = DataLoader(adv_dataset, batch_size=1)

    if check_adv_validity(ori_dataset, adv_dataset, epsilon):
        print('✓ Adversarial dataset validity passed!')
    else:
        print('❌ Adversarial dataset validity not passed!')
        return

    accuracies = np.empty((len(eval_models), 10))

    for i, model_name in enumerate(eval_models):
        model = get_model(model_name, pretrained=True)
        accuracies[i] = evaluate_single_model(model_name.replace('_cifar10', ''), model, adv_dataloader)

    means = np.mean(accuracies, axis=1)

    # Print markdown table
    with (output_dir / f'model_acc.md').open('w') as f:
        print('Evaluation results: ', file=f)
        print(f"| Models     | {' | '.join(s.replace('_cifar10', '') for s in eval_models)} |", file=f)
        print(f"| ---------- |{'|'.join('------' for _ in eval_models)}|", file=f)

        for i, label_name in enumerate(all_labels):
            print(f"| {label_name:10s} |{'|'.join(f' {k:.2f} ' for k in accuracies[:, i])}|", file=f)

        print(f"| Mean       |{'|'.join(f' {k:.2f} ' for k in means)}|", file=f)


def main():
    args = parse_arguments()

    if args.source_dir is None:
        args.source_dir = (
            args.proxy_models[0] if len(args.proxy_models) == 1 else ''.join(m[0] for m in args.proxy_models) +
            f'-{args.target_method[0]}' +
            f'-{args.num_iters}'
        )

    kwargs = vars(args)

    seed_everything(SEED)

    if args.task == 'attack':
        attack_root(**kwargs)
        evaluate_root(**kwargs)
    elif args.task == 'evaluate':
        evaluate_root(**kwargs)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('task', help='choose one from {attack, evaluate}')
    parser.add_argument('--source_dir', type=lambda p: Path(p).absolute(), help='the directory of the original validating images')
    parser.add_argument('--output_dir', type=lambda p: Path(p).absolute(), help='the directory of the output adversarial images')
    parser.add_argument('--proxy_models', nargs='+', help='proxy model for generating adversarial images')
    parser.add_argument('--epsilon', type=float, default=8 / 256, help='the l-infinity value in [0, 1], default 8/256 = 0.03125')
    parser.add_argument('--num_iters', type=int, default=1, help='number of iterations for iterative FGSM, default 1')
    parser.add_argument('--target_method', default='untargeted', help='method for target generation, choose one from {untargeted, random, next}, default negative')
    return parser.parse_args()


if __name__ == '__main__':
    main()
