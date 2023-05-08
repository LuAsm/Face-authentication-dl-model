# app.py

import io
import random
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Subset
from sklearn.preprocessing import LabelEncoder
from Face_recognition import names, modelV3
from Data_to_dataset import MyDataset1
import pickle
app = Flask(__name__)

def load_model():
    model = modelV3
    model = torch.load('ModelV3_t.pth')
    model.eval()
    return model

# Load preprocessed data from file
with open('alll_data.pickle', 'rb') as f:
    list_of_data = pickle.load(f)

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

def random_image():
    idx = random.randint(0, len(test_data) - 1)
    img, label = test_data[idx]

    return img, label


# Encode the targets
le = LabelEncoder()
dataset.labels = le.fit_transform(dataset.labels)

# Decode the targets
decoded_targets = le.inverse_transform(dataset.labels)
print("Decoded targets:", decoded_targets)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/random_image')
def get_random_image():
    img, label = random_image()
    img = (img * 255).byte().squeeze().numpy()
    img_pil = Image.fromarray(img, mode='RGB')
    
    # Resize the image to 512x512 pixels
    img_pil = img_pil.resize((512, 512), resample=Image.LANCZOS)
    
    # Load the model
    model = load_model()

      # Convert img back to tensor and perform the prediction
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    pred = model(img_tensor)
    
    # Get the predicted class name
    pred_class = names[le.inverse_transform([torch.argmax(pred, 1).item()])[0]]

    # Get the ground truth label
    truth_label = names[le.inverse_transform([label])[0]] 

    # Add text to the image
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Predicted: {pred_class} | Truth: {truth_label}", font=font, fill=(255,255,255))

    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)