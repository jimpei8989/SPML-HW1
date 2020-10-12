import torch
from tqdm import tqdm

from modules.dataset import AdversarialDataset
from modules.soft_crossentropy import SoftCrossEntropyLoss


def remove_grads(model):
    for param in model.parameters():
        param.requires_grad = False


def generate_target(label, method: str, num_classes=10):
    exclude = [i for i in range(num_classes) if i != label]
    ret = torch.zeros(num_classes)
    if method == 'negative':
        ret[exclude] = 1
    elif method == 'next':
        ret[(label + 1) % num_classes] = 1
    elif method == 'random':
        ret[exclude[torch.randint(num_classes - 1, (1,))]] = 1
    return ret.unsqueeze(0)


def fgsm_attack(model, dataloader, output_dir, epsilon, num_iters, target_method='negative'):
    remove_grads(model)

    model.cuda()
    model.eval()

    adv_dataset = AdversarialDataset(output_dir)

    criterion = SoftCrossEntropyLoss()

    for image_name, image, label in tqdm(dataloader, desc='Attacking'):
        target = generate_target(label, method=target_method).cuda()

        upper_bound = torch.min(torch.ones_like(image), image + epsilon)
        lower_bound = torch.max(torch.zeros_like(image), image - epsilon)

        history = []

        for _ in range(num_iters):
            image.requires_grad = True
            logits = model(image)
            loss = criterion(logits, target)

            loss.backward()

            grad = image.grad.data.sign()

            image.requires_grad = False

            image += epsilon * grad
            image = torch.max(torch.min(image, upper_bound), lower_bound)

            with torch.no_grad():
                val_loss = criterion(logits, target).item()
                history.append((image.clone(), val_loss))

        image, loss = min(history, key=lambda p: p[1])
        adv_dataset.add(image_name, image, label)

    return adv_dataset
