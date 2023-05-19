import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import glob
from src.trained_model_load_for_update import  modelV3 


# Step 1: Preprocess the new images
def preprocess_images(folder_path, label, num_augmentations=4):
    image_paths = glob.glob(folder_path + "/*.jpg")
    images = []
    labels = []

    # Define the augmentation transforms
    augmentation_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for path in image_paths:
        # Load the image using PIL or OpenCV
        image = Image.open(path)  # Using PIL
        # image = cv2.imread(path)  # Using OpenCV

        # Apply the augmentation transforms to the original image
        augmented_image = augmentation_transforms(image)

        images.append(augmented_image)
        labels.append(label)

        # Generate additional augmented images
        for _ in range(num_augmentations - 1):
            augmented_image = augmentation_transforms(image)
            images.append(augmented_image)
            labels.append(label)

    # Stack the images into tensors
    images = torch.stack(images)

    return images, labels

# Step 2: Create a new dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        return image, label
    
# Step 3: Load trained model

load_model = modelV3

# Step 4: Train the model
def train_model(model, dataset, num_epochs, batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss}")
    
    return model

# Step 5: Evaluate the model
def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += 1
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    return accuracy

# Step 6: Save the updated model
def save_model(model, model_path):
    
    torch.save(model.state_dict(),
            model_path)
    print("Model saved successfully.")




folder_path = "uploads"  # Path to the directory

# Get the list of items (files and folders) within the directory
items = os.listdir(folder_path)

# Iterate over the items and find the first folder
for item in items:
    item_path = os.path.join(folder_path, item)
    if os.path.isdir(item_path):
        # Found the first folder
        first_folder = item_path
        break


label = 159
    # Step 1: Preprocess the new images
preprocessed_images, labels = preprocess_images(first_folder, label)
    
    # Step 2: Create a new dataset
dataset = CustomDataset(preprocessed_images, labels)

    # Step 3: Load trained model
model = load_model

# Step 4: Train the model
num_epochs = 5
batch_size = 2
trained_model = train_model(model, dataset, num_epochs, batch_size)
    
# Step 5: Evaluate the model
evaluate_model(trained_model, dataset)
    
# Step 6: Save the updated model
save_model(trained_model, "updated_model.pth")
    
    
