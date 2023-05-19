from torch.utils.data import Dataset
import torch
from torch import nn 
import random, pickle
from torch.utils.data import Subset
from sklearn.preprocessing import LabelEncoder

import os

class MyDataset1(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

DATA_NAME = 'alll_data.pickle'

if os.path.exists(DATA_NAME):
    # Load the model state dict from the file
    print(f"Loading data from: {DATA_NAME}")
    # Load preprocessed data from file
    with open('alll_data.pickle', 'rb') as f:
        list_of_data = pickle.load(f)
else:
    # Train the model and save the state dict
    print("Making dataset... You can go get a coffee, it will take about (10-15 min)")
    from make_data import alll_data
    list_of_data = alll_data


images = [x[0] for x in list_of_data]
labels = [x[1] for x in list_of_data]

dataset = MyDataset1(images, labels)

# Split the dataset into training and test sets
n_samples = len(dataset)
train_size = int(0.8 * n_samples)

test_size = n_samples - train_size

indices = list(range(n_samples))
random.shuffle(indices)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

#load names
with open('./data/lfw-py/lfw-names.txt') as f:
    
    names = [name.split("\t")[0] for name in f.read().splitlines()]
# Encode the targets
le = LabelEncoder()
dataset.labels = le.fit_transform(dataset.labels)