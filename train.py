import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import plot_training_validation

def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_loader = load_dataset(data_path=args.data_path, mode="train", batch_size=args.batch_size, num_workers=4)
    val_loader = load_dataset(data_path=args.data_path, mode="valid", batch_size=args.batch_size, num_workers=4)

    # initialize model, loss, and optimizer
    model = UNet(n_channels=3, n_classes=1).to(device) # Unet/ResNet34_UNet
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    val_scores = []
    train_scores = []

    # train
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device) 
            masks = batch['mask'].to(device) 

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {avg_train_loss:.4f}")

        train_dice = evaluate(model, train_loader, device)
        train_scores.append(train_dice)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Training Dice Score: {train_dice:.4f}")

        val_dice = evaluate(model, val_loader, device)
        val_scores.append(val_dice)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Dice Score: {val_dice:.4f}")

    # save model
    os.makedirs("../saved_models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("../saved_models/checkpoints", "unet_model.pth")) # unet_model.pth/resnet34_unet_model.pth
    print("Model saved to ../saved_models/checkpoints/unet_model.pth") # resnet34_unet_model.pth

    # plot training and validation trends
    plot_training_validation(train_losses, train_scores, val_scores, "UNet", save_path="./plots/plot.png") # UNet/ResNet34_UNet

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)


# Unet
# python train.py --data_path ../dataset/Oxford_IIIT_Pet --epochs 10 --batch_size 8 --learning-rate 1e-4
# modify: 21, 59, 60, 63
# ResNet34_UNet
# python train.py --data_path ../dataset/Oxford_IIIT_Pet --epochs 10 --batch_size 8 --learning-rate 1e-4