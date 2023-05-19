import pickle
import torch
import os
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import random

#load data


all_data = datasets.LFWPeople(
    root="/home/empar/data", 
    split = "10fold", 
    transform=ToTensor(), 
    target_transform=None, 
    download=True 
)

#load names
with open('./data/lfw-py/lfw-names.txt') as f:
    
    names = [name.split("\t")[0] for name in f.read().splitlines()]

def fixdata():
    identity_counts = [list(all_data.targets).count(i) for i in range(len(set(all_data.targets)))]

        # Iterate through the dataset in reverse order and remove the samples with less than 10 images
    for i in reversed(range(len(all_data))):
        if identity_counts[all_data.targets[i]] < 10:
            all_data.data.pop(i)
            all_data.targets.pop(i)
    
    for _ in range(0,6):
    # Convert the list of targets to a Tensor
        targets = torch.tensor(all_data.targets)
        # Count the number of images per label
        counts = torch.bincount(targets)

        # Iterate through the dataset and remove images from labels with more than 50 images

        for i, (image, label) in enumerate(all_data):
            if counts[label] > 50:
                if random.random() > 50 / counts[label]:
                # Remove the image and update the counts
                    all_data.data.pop(i)
                    all_data.targets.pop(i)
                    counts[label] -= 1

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def createnewimg():
    # Create a dictionary to store the counts for each label
    label_counts = {}

    # Loop through the dataset and count the number of images for each label
    for i in range(len(all_data)):
        _, label = all_data[i]
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

# Find all labels with less than 40 images
    labels_to_augment = [label for label, count in label_counts.items() if count < 40]

# Define the transforms to use for the image augmentation
    rotation_transform = transforms.RandomRotation(degrees=15)
    horizontal_flip_transform = transforms.RandomHorizontalFlip()
    color_jitter_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    resize_crop_transform = transforms.RandomResizedCrop(size=(250, 250), scale=(0.8, 1.0))
    gaussian_noise_transform = transforms.Lambda(lambda x: GaussianNoise(mean=0, std=0.1)(x))

# Create a new list to store all the data, including augmented images
    alll_data = list(all_data)

# Loop through the labels to augment
    for label in labels_to_augment:
    # Create a list of indices for all images with the current label
        label_indices = [i for i in range(len(all_data)) if all_data[i][1] == label]
    # Loop until the label has 40 images
        while label_counts[label] < 40:
        # Choose a random image index to augment
            image_index = label_indices[torch.randint(len(label_indices), size=(1,)).item()]
        # Load the image and apply a random augmentation
            image, _ = all_data[image_index]
            transform_choice = torch.randint(5, size=(1,)).item()
            if transform_choice == 0:
                image = rotation_transform(image)
            elif transform_choice == 1:
                image = horizontal_flip_transform(image)
            elif transform_choice == 2:
                image = color_jitter_transform(image)
            elif transform_choice == 3:
                image = resize_crop_transform(image)
            elif transform_choice == 4:
                image = gaussian_noise_transform(image)
        # Add the augmented image to the dataset
            alll_data.append((image, label))
        # Update the label count
            label_counts[label] += 1
        

    return alll_data

# Check if the model file exists
DATA_NAME = 'alll_data.pickle'


print("start fix")
fixdata()
alll_data = createnewimg()
# Save the model state dict
print(f"Saving model to: {DATA_NAME}")
with open(DATA_NAME, 'wb') as f:
    pickle.dump(alll_data, f)