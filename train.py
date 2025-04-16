from tqdm import tqdm 
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, pad_token_id, epochs=30, patience=5, device="cpu"):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", unit="batch") as tepoch:
            for batch in tepoch:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop with progress bar
        model.eval()
        val_loss = 0.0
        with tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)", unit="batch") as vepoch:
            with torch.no_grad():
                for batch in vepoch:
                    inputs = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    val_loss += loss.item()
                    vepoch.set_postfix(loss=loss.item())
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

        scheduler.step(val_loss)
        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    save_loss_curves(model.__class__.__name__, train_losses, val_losses)












import os
import matplotlib.pyplot as plt

def save_loss_curves(model_name, train_losses, val_losses, save_dir="."):
    """
    Save training and validation loss curves as a PNG file.
    Args:
        model_name (str): Name of the model (e.g., "RNN", "LSTM", "Transformer").
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        save_dir (str): Directory to save the loss curve plot.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title(f"{model_name} Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_path = os.path.join(save_dir, f"{model_name}_loss_curve.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved loss curve: {save_path}")