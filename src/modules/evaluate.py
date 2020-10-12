import numpy as np
import torch

from tqdm import tqdm


def check_adv_validity(ori_dataset, adv_dataset, epsilon):
    for i, (a_img, b_img) in enumerate(
        zip(ori_dataset.get_np_images(), adv_dataset.get_np_images())
    ):
        if np.any(np.abs(a_img - b_img) > epsilon * 256):
            print(i, np.max(np.abs(a_img - b_img)))
            return False
    return True


def evaluate_single_model(model_name, model, dataloader):
    model.cuda()
    model.eval()

    ret = np.zeros(10)

    with torch.no_grad():
        for image_name, image, label in tqdm(dataloader, desc=f'Evaluating {model_name[:20]:20s}'):
            output_logits = model(image).cpu()
            output_label = torch.argmax(output_logits, dim=1)
            ret[label] += int(output_label == label.cpu())

    return ret / 10
