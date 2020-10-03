import numpy as np
import torch

from tqdm import tqdm

def evaluate_single_model(model, dataloader):
    model.cuda()
    model.eval()

    ret = np.zeros(10)

    for image_name, image, label in tqdm(dataloader, desc='Evaluating'):
        output_logits = model(image).cpu()
        output_label = torch.argmax(output_logits, dim=1)

        ret[label] += int(output_label == label)

    return ret / 10
