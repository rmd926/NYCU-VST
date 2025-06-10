import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=100):
        super(ClassificationModel, self).__init__()
        
        # Reduced number of filters in convolutional layers
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)  # 3 filters
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)  # 6 filters
        self.conv3 = nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1)  # 8 filters

        self.pool = nn.MaxPool2d(4, 4)  # Pooling to reduce the spatial dimensions by a factor of 4
        self.dropout = nn.Dropout(0.15)  # Dropout to reduce overfitting (15% dropout rate)
        
        # Fully connected layer size is adjusted for the output size of the conv3 layer (8 channels, 3x3)
        self.fc1 = nn.Linear(8 * 3 * 3, num_classes)  # 8 filters with 3x3 feature map size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # After conv1 -> ReLU -> pool (224x224 -> 56x56)
        x = self.pool(F.relu(self.conv2(x)))  # After conv2 -> ReLU -> pool (56x56 -> 14x14)
        x = self.pool(F.relu(self.conv3(x)))  # After conv3 -> ReLU -> pool (14x14 -> 3x3)
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 8 * 3 * 3)  # Flatten the tensor (8 channels, 3x3 feature map size)
        
        x = self.dropout(x)  # Apply dropout to reduce overfitting
        x = self.fc1(x)  # Fully connected layer to get class scores
        
        return x
