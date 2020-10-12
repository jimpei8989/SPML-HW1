from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model

from modules.dataset import OriginalDataset
from modules.fgsm import fgsm_attack
from modules.evaluate import check_adv_validity, evaluate_single_model
from modules.utils import all_labels, eval_models, target_models

SEED = 0x06902029


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def attack_root(target_model, source_dir, output_dir, **kwargs):
    if not target_model.endswith('_cifar10'):
        target_model += '_cifar10'

    model = get_model(target_model, pretrained=True)

    ori_dataset = OriginalDataset(source_dir)
    ori_dataloader = DataLoader(ori_dataset, batch_size=1)
    adv_dataset = fgsm_attack(model, ori_dataloader, output_dir, **kwargs)
    adv_dataset.save_to_directory()


def evaluate_root(source_dir, output_dir, epsilon, eval_mode, **kwargs):
    models = {
        'all': eval_models,
        'fast': target_models,
    }[eval_mode]

    ori_dataset = OriginalDataset(source_dir)
    adv_dataset = OriginalDataset(output_dir)
    adv_dataloader = DataLoader(adv_dataset, batch_size=1)

    if check_adv_validity(ori_dataset, adv_dataset, epsilon):
        print('✓ Adversarial dataset validity passed!')
    else:
        print('❌ Adversarial dataset validity not passed!')
        return

    accuracies = np.empty((len(models), 10))

    for i, model_name in enumerate(models):
        model = get_model(model_name, pretrained=True)
        accuracies[i] = evaluate_single_model(model_name.replace('_cifar10', ''), model, adv_dataloader)

    means = np.mean(accuracies, axis=1)

    # Print markdown table
    with (Path(output_dir) / f'model_acc_{eval_mode}.md').open('w') as f:
        print('Evaluation results: ', file=f)
        print(f"| Models     | {' | '.join(s.replace('_cifar10', '') for s in models)} |", file=f)
        print(f"| ---------- |{'|'.join('------' for _ in models)}|", file=f)

        for i, label_name in enumerate(all_labels):
            print(f"| {label_name:10s} |{'|'.join(f' {k:.2f} ' for k in accuracies[:, i])}|", file=f)

        print(f"| Mean       |{'|'.join(f' {k:.2f} ' for k in means)}|", file=f)


def main():
    args = parse_arguments()
    kwargs = vars(args)

    seed_everything(SEED)

    if args.task == 'attack':
        attack_root(**kwargs)
        evaluate_root(**kwargs)
    elif args.task == 'evaluate':
        evaluate_root(**kwargs)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--source_dir', type=lambda p: Path(p).absolute())
    parser.add_argument('--output_dir', type=lambda p: Path(p).absolute())
    parser.add_argument('--target_model')
    parser.add_argument('--epsilon', type=float, default=8 / 256)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--target_method', default='random')
    parser.add_argument('--eval_mode', default='fast')   # fast, for targeted models only; all for all models
    return parser.parse_args()


if __name__ == '__main__':
    main()
