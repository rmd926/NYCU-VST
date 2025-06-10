import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import get_train_val_loaders
from model import ClassificationModel
from tqdm import tqdm  # Progress bar for displaying the training process

# Set dataset path and batch size
student_id = '313553024'
data_dir = r'C:\Users\user\Desktop\hw1_313553024'
batch_size = 32

# Check if GPU is available; if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Get DataLoader for training and validation sets
train_loader, val_loader = get_train_val_loaders(data_dir, batch_size)

# Initialize model and move it to GPU (if available)
model = ClassificationModel(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)# 0.0005

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Early stopping parameters
early_stopping_patience = 10  # Number of epochs to wait before stopping
early_stopping_counter = 0
best_val_acc = 0.0  # Initialize best validation accuracy

# Training loop
for epoch in range(100):  # Train for up to 50 epochs
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Display progress bar using tqdm
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{100}', unit="batch") as tepoch:
        for inputs, labels in tepoch:
            # Move inputs and labels to GPU (if available)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Top-1 accuracy
            _, predicted_top1 = outputs.max(1)
            total += labels.size(0)
            correct_top1 += predicted_top1.eq(labels).sum().item()

            # Top-5 accuracy
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            correct_top5 += sum([labels[i] in predicted_top5[i] for i in range(len(labels))])

            running_loss += loss.item()

            # Update tqdm with loss and accuracy
            tepoch.set_postfix(loss=running_loss/len(train_loader), top1_acc=correct_top1/total, top5_acc=correct_top5/total)

    # Validation
    model.eval()
    val_correct_top1 = 0
    val_correct_top5 = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move validation data to GPU (if available)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted_top1 = outputs.max(1)
            val_total += labels.size(0)
            val_correct_top1 += predicted_top1.eq(labels).sum().item()

            # Top-5 accuracy
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            val_correct_top5 += sum([labels[i] in predicted_top5[i] for i in range(len(labels))])

    val_top1_acc = 100. * val_correct_top1 / val_total
    val_top5_acc = 100. * val_correct_top5 / val_total
    print(f"Validation Top-1 Accuracy: {val_top1_acc}%, Top-5 Accuracy: {val_top5_acc}%")

    # Update learning rate based on validation Top-1 accuracy
    scheduler.step(val_top5_acc)

    # Save the model if the validation Top-1 accuracy improves
    if val_top5_acc > best_val_acc:
        best_val_acc = val_top5_acc
        torch.save(model.state_dict(), f'w_{student_id}.pth')  # Save best model weights
        print(f"Model saved with Top-5 Validation Accuracy: {best_val_acc}%")
        early_stopping_counter = 0  # Reset early stopping counter
    else:
        early_stopping_counter += 1

    # Check for early stopping
    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# After training, the best model will be saved as 'w_{student_id}_best.pth'
