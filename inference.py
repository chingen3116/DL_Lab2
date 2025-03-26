import argparse
import os
import torch
import numpy as np
from PIL import Image
from models.unet import UNet 
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import visualize_segmentation

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--output_path', type=str, default='./output', help='path to save the predicted masks')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

def load_model(model_path):
    model = UNet(n_channels=3, n_classes=1) # ResNet34_UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def save_mask(mask, output_path, original_image):
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image = mask_image.resize(original_image.size)  
    mask_image.save(output_path)

if __name__ == '__main__':
    args = get_args()
    model, device = load_model(args.model)

    # load test data
    test_loader = load_dataset(data_path=args.data_path, mode="test", batch_size=args.batch_size, num_workers=4)

    os.makedirs(args.output_path, exist_ok=True)
    # Calculate average Dice Score
    dice_score = evaluate(model, test_loader, device=device)
    print(f"Average Dice Score on test set: {dice_score:.4f}")

    # visualization and save results
    for i, batch in enumerate(test_loader):
        if i == 50:
            break

        print(f"Processing {i+1}/{len(test_loader)}")
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)  # Ground truth masks
        filenames = batch["filename"]  # File names for saving results

        with torch.no_grad():
            outputs = model(images)
            pred_masks = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)

        for j in range(len(pred_masks)):
            pred_mask = (pred_masks[j] > 0.5).cpu().numpy().astype(np.uint8)  # Binarize predicted mask
            gt_mask = masks[j].cpu()  # Ground truth mask
            image = images[j]  # Original image tensor
            filename = filenames[j]

            # Save predicted mask
            mask_path = os.path.join(args.output_path, f"{os.path.splitext(filename)[0]}_mask.png") # _res
            save_mask(pred_mask, mask_path, Image.open(os.path.join(args.data_path, "images", f"{filename}.jpg")))
            print(f"Predicted mask saved to {mask_path}")

            # Visualize segmentation
            vis_path = os.path.join(args.output_path, f"{os.path.splitext(filename)[0]}_visualization.png") # _res
            visualize_segmentation(image, pred_mask, gt_mask, save_path=vis_path)
            print(f"Visualization saved to {vis_path}")

    print(f"Predicted masks and visualizations saved to {args.output_path}")


# UNet
# python inference.py --model ../saved_models/checkpoints/unet_model.pth --data_path ../dataset/Oxford_IIIT_Pet --batch_size 4
# modify: 22, 67, 72
# ResNet34_UNet
# python inference.py --model ../saved_models/checkpoints/resnet34_unet_model.pth --data_path ../dataset/Oxford_IIIT_Pet --batch_size 4