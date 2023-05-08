# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Subset
from sklearn.preprocessing import LabelEncoder
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
# Import matplotlib for visualization
import matplotlib.pyplot as plt
import numpy as np
# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldnt' be lower than 0.11
print(f'PyTorch version: {torch.__version__}\ntorchvision version: { torchvision.__version__}')

from collections import Counter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
from collections import defaultdict
import pandas as pd
import os
from timeit import default_timer as timer 
import pickle

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

# Find all labels with less than 42 images
    labels_to_augment = [label for label, count in label_counts.items() if count < 42]

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
    # Loop until the label has 42 images
        while label_counts[label] < 42:
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

if os.path.exists(DATA_NAME):
    # Load the model state dict from the file
    print(f"Loading model from: {DATA_NAME}")
    with open(DATA_NAME, 'rb') as f:
        list_of_data = pickle.load(f)
else:
    print("start fix")
    fixdata()
    alll_data = createnewimg()
    # Save the model state dict
    print(f"Saving model to: {DATA_NAME}")
    with open(DATA_NAME, 'wb') as f:
        pickle.dump(alll_data, f)





images = [x[0] for x in list_of_data]
labels = [x[1] for x in list_of_data]
#Create new dataset with created images

class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

new_dataset = MyDataset(images, labels)



# Encode the targets
le = LabelEncoder()
new_dataset.labels = le.fit_transform(new_dataset.labels)

# Decode the targets
decoded_targets = le.inverse_transform(new_dataset.labels)
print("Decoded targets:", decoded_targets)


# Split the dataset into training and test sets
n_samples = len(new_dataset)
train_size = int(0.8 * n_samples)

test_size = n_samples - train_size

indices = list(range(n_samples))
random.shuffle(indices)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_data = Subset(new_dataset, train_indices)
test_data = Subset(new_dataset, test_indices)

# Setup the batch size hyperparamete
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
      batch_size = BATCH_SIZE, # how many sample per batch?
      shuffle = True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data, 
      batch_size = BATCH_SIZE, 
      shuffle = False # don't necessarily have shuffle the testing data
)

device = "cuda" if torch.cuda.is_available() else "cpu"
device

torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader,device: torch.device = device):
   
    loss, acc = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device) 

            # Make predictions with the model
            y_pred = model(images)
            
            log_probabilities = model(images)
            loss += criterion(log_probabilities, labels).item()

            probabilites = torch.exp(log_probabilities)
            top_prob, top_class = probabilites.topk(1, dim=1)
            predictions = top_class == labels.view(*top_class.shape)
            acc += torch.mean(predictions.type(torch.FloatTensor))
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss,
            "model_acc": acc}

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

class DeepID(nn.Module):
    def __init__(self):
        super(DeepID, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 20, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(20)
        self.rel1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolutional block
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(40)
        self.rel2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional block
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(60)
        self.rel3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc4 = nn.Linear(60 * 29 * 29, 160)
        self.bn4 = nn.BatchNorm1d(160)
        self.rel4 = nn.ReLU()
        self.fc5 = nn.Linear(160, out_features=len(set(new_dataset.labels)))

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rel1(x)
        x = self.pool1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rel2(x)
        x = self.pool2(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rel3(x)
        x = self.pool3(x)

        # Flatten the output from convolutional blocks
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.rel4(x)

        x = self.fc5(x)

        return x
    
modelV3 = DeepID()
modelV3.to(device)

optimizer = optim.Adam(modelV3.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Measure time
from timeit import default_timer as timer
train_time_start_model_3 = timer()

epochs = 5 # The total number of iteration

train_lossesV3 = []
test_lossesV3 = []

for epoch in range(epochs):
  # model for training
  modelV3.train()
  train_loss = 0

  for idx, (images, labels) in enumerate(train_dataloader):

    # Send these >>> To GPU
    images = images.to(device)
    labels = labels.to(device)

    # Training pass
    optimizer.zero_grad()

    # Forward pass
    output = modelV3(images)
    loss = criterion(output, labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  else:
    # model for evaluation
    modelV3.eval()
    test_loss = 0
    accuracy = 0

    # Turn off gradients when performing evaluation.
    # As if we don't turn it off, we  will comprise our networks weight entirely
    with torch.no_grad():
      for images, labels in test_dataloader:

        images = images.to(device)
        labels = labels.to(device) 

        log_probabilities = modelV3(images)
        test_loss += criterion(log_probabilities, labels).item()

        probabilites = torch.exp(log_probabilities)
        top_prob, top_class = probabilites.topk(1, dim=1)
        predictions = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(predictions.type(torch.FloatTensor))

      train_lossesV3.append(train_loss/len(train_dataloader))
      test_lossesV3.append(test_loss/len(train_dataloader))

      print("Epoch: {}/{}   ".format(epoch+1, epochs),
            "Training loss: {:.4f}   ".format(train_loss/len(train_dataloader)),
            "Testing loss: {:.4f}   ".format(test_loss/len(train_dataloader)),
            "Test accuracy: {:.4f}   ".format(accuracy/len(test_dataloader)))
      
train_time_end_model_3 = timer()
total_train_time_model_3 = print_train_time(start=train_time_start_model_3,
                                           end=train_time_end_model_3,
                                           device=device)

# Calculate model 1 results
model_3_results = eval_model(model=modelV3, data_loader=test_dataloader,
    device=device
)
model_3_results

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    modelV3.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# Make predictions on test samples with model 2
pred_probs = make_predictions(model=modelV3,
                              data=test_samples)

# Turn the prediction probabilities into prediction labells by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)
pred_classes

# Plot predictions
plt.figure(figsize=(16, 16))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create a subplot
    plt.subplot(nrows, ncols, i+1)

    # Plot the target image
    plt.imshow(sample.permute(1,2,0))

    # Find the prediction label (in text form, e.g. "Sandal")
    pred_label = names[le.inverse_transform([pred_classes[i]])[0]]

    # Get the truth label (in text form, e.g. "T-shirt")
    truth_label = names[le.inverse_transform([test_labels[i]])[0]] 

    # Create the title text of the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Check for equality and change title colour accordingly
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g") # green text if correct
    else:
        plt.title(title_text, fontsize=10, c="r") # red text if wrong

    # Remove tick marks from the axes
    plt.xticks([], [])
    plt.yticks([], [])

    # Hide axis labels
    plt.axis(False)

# 10. Save and load the model

# Create model save path
MODEL_NAME = "ModelV3_t.pth"

# Check if the model file exists
if os.path.exists(MODEL_NAME):
    # Load the model state dict from the file
    print(f"Loading model from: {MODEL_NAME}")
    modelV3.load_state_dict(torch.load(MODEL_NAME))
else:
    # Save the model state dict
    print(f"Saving model to: {MODEL_NAME}")
    torch.save(obj=modelV3.state_dict(),
               f=MODEL_NAME)

