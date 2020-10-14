import numpy as np
import torch

from tqdm import tqdm

from modules.dataset import cifar10_mean, cifar10_std
from modules.normalize import Normalize


def check_adv_validity(ori_dataset, adv_dataset, epsilon):
    for i, (a_img, b_img) in enumerate(
        zip(ori_dataset.get_np_images(), adv_dataset.get_np_images())
    ):
        if np.any(np.abs(a_img - b_img) > epsilon * 256):
            print(i, np.max(np.abs(a_img - b_img)))
            return False
    return True


def evaluate_single_model(model_name, model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = Normalize(cifar10_mean, cifar10_std).to(device)
    model.to(device)
    model.eval()

    ret = np.zeros(10)

    with torch.no_grad():
        for _, image, label in tqdm(dataloader, desc=f'Evaluating {model_name[:20]:20s}'):
            output_logits = model(normalize(image.to(device))).cpu()
            output_label = torch.argmax(output_logits, dim=1)
            ret[label] += int(output_label == label)

    return ret / 10
