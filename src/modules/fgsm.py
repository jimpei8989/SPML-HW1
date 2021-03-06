from collections.abc import Iterable

import torch
from torch.nn import CrossEntropyLoss
from torchvision.transforms import functional as TF
from tqdm import tqdm

from modules.dataset import AdversarialDataset, cifar10_mean, cifar10_std
from modules.normalize import Normalize
from modules.soft_crossentropy import SoftCrossEntropyLoss


def remove_grads(model):
    for param in model.parameters():
        param.requires_grad = False


def generate_target(label, method: str, num_classes=10):
    exclude = [i for i in range(num_classes) if i != label]
    ret = torch.zeros(num_classes)
    if method == 'untargeted':
        ret[label] = -1
    elif method == 'next':
        ret[(label + 1) % num_classes] = 1
    elif method == 'random':
        ret[exclude[torch.randint(num_classes - 1, (1,))]] = 1
    else:
        raise ValueError('method should be in {untargeted, next, random}')
    return ret.unsqueeze(0)


def attack(model, dataloader, output_dir, epsilon=0.03125, step_size=None, num_iters=1, decay_factor=0.0, target_method='untargeted', **kwargs):
    if step_size is None or step_size == 'same':
        step_size = epsilon
    elif step_size == 'divide':
        step_size = epsilon / num_iters

    if not isinstance(decay_factor, Iterable):
        decay_factor = [decay_factor]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = Normalize(cifar10_mean, cifar10_std).to(device)

    remove_grads(model)

    model.to(device)
    model.eval()

    adv_dataset = AdversarialDataset(output_dir)

    criterion = SoftCrossEntropyLoss()
    val_criterion = CrossEntropyLoss()

    for image_name, ori_image, ori_label in tqdm(dataloader, desc='Attacking'):
        history = []

        for d in decay_factor:
            image = ori_image.clone().to(device)
            label = ori_label.clone().to(device)
            target = generate_target(label, method=target_method).to(device)

            upper_bound = torch.min(torch.ones_like(image), image + epsilon)
            lower_bound = torch.max(torch.zeros_like(image), image - epsilon)

            g = None

            for _ in range(num_iters):
                image.requires_grad = True

                logits = model(normalize(image))
                loss = criterion(logits, target)

                loss.backward()

                grad = image.grad.detach().sign()
                if g is None:
                    g = grad
                else:
                    g = d * g + grad

                image.requires_grad = False
                image -= step_size * g.sign()

                # Clip image into [0, 1] and with in original image +/- epsilon
                image = torch.max(torch.min(image, upper_bound), lower_bound)

                with torch.no_grad():
                    logits = model(normalize(image))
                    val_loss = val_criterion(logits, label).item()
                    history.append((image.cpu(), val_loss))

        image, loss = max(history, key=lambda p: p[1])
        adv_dataset.add(image_name, image, label)

    return adv_dataset
