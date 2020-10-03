import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.dataset import AdversarialDataset
from modules.soft_crossentropy import SoftCrossEntropyLoss

def remove_grads(model):
    for param in model.parameters():
        param.requires_grad = False


def generate_target(label, method: str, num_classes = 10):
    exclude = [i for i in range(num_classes) if i != label]
    ret = torch.zeros(num_classes)
    if method == 'negative':
        ret[exclude] = 1
    elif method == 'next':
        ret[(label + 1) % num_classes] = 1
    elif method == 'random':
        ret[exclude[torch.randint(num_classes - 1, (1,))]] = 1
    return ret.unsqueeze(0)


def fgsm_attack(model, dataloader, output_dir, epsilon, target_method='negative'):
    remove_grads(model)

    model.cuda()
    model.eval()

    adv_dataset = AdversarialDataset(output_dir)

    criterion = SoftCrossEntropyLoss()

    for image_name, image, label in tqdm(dataloader, desc='Attacking'):
        image.requires_grad = True

        target = generate_target(label, method=target_method).cuda()

        logits = model(image)
        loss = criterion(logits, target)

        loss.backward()

        perturbed_image = torch.clamp(image + epsilon * image.grad.data.sign(), 0, 1)

        adv_dataset.add(image_name, perturbed_image, label)

    return adv_dataset
