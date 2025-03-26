import torch
import numpy as np
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask, epsilon=1e-6):
    # implement the Dice score here
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    intersection = torch.sum(pred_mask * gt_mask) 
    union = torch.sum(pred_mask) + torch.sum(gt_mask) 

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice.item()

'''
pred = torch.tensor([[0, 1, 1], [1, 1, 0], [0, 1, 1]])
gt = torch.tensor([[0, 1, 0], [1, 1, 0], [0, 1, 1]])
print(dice_score(pred, gt))
'''

def visualize_segmentation(image, pred_mask, gt_mask, save_path=None):
    # if tensors, convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy() if image.ndim == 3 else image.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # Remove extra channel dimension if present
    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = np.squeeze(pred_mask, axis=0)
    if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
        gt_mask = np.squeeze(gt_mask, axis=0)

    # Normalize image for visualization
    image = (image - image.min()) / (image.max() - image.min())

    # Create a figure
    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.title("Original Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.imshow(gt_mask, alpha=0.5, cmap='jet')  # Overlay ground truth mask
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.imshow(pred_mask, alpha=0.5, cmap='jet')  # Overlay predicted mask
    plt.title("Predicted Mask")
    plt.axis("off")

    # Save or show the visualization
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt

def plot_training_validation(train_losses, train_scores, val_scores, model_name, save_path=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o', color='blue')
    plt.plot(epochs, val_scores, label="Validation Dice Score", marker='x', color='orange')
    plt.plot(epochs, train_scores, label="Training Dice Score", marker='x', color='green')

    plt.title(f"Training and Validation Trends ({model_name})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Dice Score")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()