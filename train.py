import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_data = datasets.ImageFolder(
    root='Project_Dataset/seg_train/seg_train',
    transform=transform
)

test_data = datasets.ImageFolder(
    root='Project_Dataset/seg_test/seg_test',
    transform=transform
)

# Data loaders
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

# Load pretrained model
model = models.resnet18(pretrained=True)

# Modify final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 6)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

# Training loop
epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Save model
torch.save(model.state_dict(), "model.pth")

print("Model saved successfully.")