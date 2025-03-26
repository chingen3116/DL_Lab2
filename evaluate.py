import torch
import numpy as np
from utils import dice_score

def evaluate(net, data, device):
    # implement the evaluation function here
    net.eval()  # evaluation mode
    dice_scores = []

    with torch.no_grad(): 
        for batch in data:
            images = batch['image'].to(device)
            true_masks = batch['mask'].to(device)

            # forward pass
            outputs = net(images)
            preds = torch.sigmoid(outputs) > 0.5

            # calculate Dice Score for each image in the batch
            for pred, gt in zip(preds, true_masks):
                dice_scores.append(dice_score(pred, gt))

    return np.mean(dice_scores)