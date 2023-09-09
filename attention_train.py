import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

import AttentionUNET
from AttentionUNET import AttU_Net
from custom_dataset import CustomDataset
import numpy as np

# Define the batch size, learning rate, number of epochs, and device to train on
batch_size = 8
learning_rate = 1e-4
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create an instance of the Attention Unet model
model = AttU_Net(in_channels=3, out_channels=1).to(device)

# Create an instance of the custom dataset for training and validation
train_dataset = CustomDataset(image_dir='./DSB/images', mask_dir='./DSB/masks',
                              transform=transforms.ToTensor())
valid_dataset = CustomDataset(image_dir='./new_data/images', mask_dir='./new_data/masks',
                              transform=transforms.ToTensor())

# Create data loaders for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    # Train the model on the training dataset
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

          # assume images is a numpy array
        transform = transforms.ToTensor()
        tensor_images=transform(images)
        ndarray_image = tensor_images.numpy()

        outputs = AttentionUNET.AttU_Net.forward(ndarray_image)  # pass tensor to model.forward() method

        optimizer.zero_grad()
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    # Evaluate the model on the validation dataset
    model.eval()
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            valid_loss += loss.item() * images.size(0)
        valid_loss /= len(valid_loader.dataset)

    # Print the training and validation loss for each epoch
    print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss, valid_loss))

# Save the trained model
torch.save(model.state_dict(), 'attention_model.h5')
