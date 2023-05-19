import torch
from torch import nn, optim
import os 


device = "cuda" if torch.cuda.is_available() else "cpu"
device

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
        self.fc5 = nn.Linear(160, out_features=158)

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

# Define the model architecture
modelV3 = DeepID()
modelV3.to(device)



# Define the optimizer and loss function
optimizer = optim.Adam(modelV3.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
    
# Check if the model file exists
MODEL_NAME = "ModelV3_org.pth"

if os.path.exists(MODEL_NAME):
    # Load the model state dict from the file
    print(f"Loading model from: {MODEL_NAME}")
    modelV3.load_state_dict(torch.load(MODEL_NAME))
    
else:
    # Train the model and save the state dict
    print("Training model...")
    from src.make_model import modelV3



modelV3.eval()