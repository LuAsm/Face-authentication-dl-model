import os
import random
import torch
from src.model_load import  device
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil

def random_image(data):
    idx = random.randint(0, len(data) - 1)
    img, label = data[idx]

    return img, label

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
        pred_probs = []
        model.eval()
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

def preprocess_image(image, label=159):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Resize the image while maintaining the aspect ratio
    width, height = image_array.shape[1], image_array.shape[0]
    aspect_ratio = width / height
    target_width = 250
    target_height = 250
    resized_image = Image.fromarray(image_array).resize((target_width, target_height))


    # Perform preprocessing steps on the resized image array
    preprocessed_image = np.array(resized_image).astype(np.float32) / 255.0
    
    # Convert the preprocessed image to a PyTorch tensor
    preprocessed_image_tensor = torch.from_numpy(preprocessed_image)
    preprocessed_image_tensor = preprocessed_image_tensor.permute(2, 0, 1)  # Rearrange dimensions (HWC to CHW)

    return preprocessed_image_tensor, label

def draw_picture(img, pred_class, truth_label):
     

     # Convert tensor to numpy array
    array = img.detach().cpu().numpy()

    # Scale the array to the 0-255 range
    scaled_array = (np.transpose(array, (1,2,0)) * 255).astype(np.uint8)

    # Create PIL Image object
    img_pil = Image.fromarray(scaled_array, mode='RGB')

    # Resize image
    img_pil = img_pil.resize((512, 512), resample=Image.LANCZOS)

        # Create a new image with the same size as the original image
    new_img = Image.new("RGB", (img_pil.width, img_pil.height + 50), (255, 255, 255))

    # Draw the original image onto the new image
    new_img.paste(img_pil, (0, 50))

    # Draw the text on top of the new image
    draw = ImageDraw.Draw(new_img)

# Add text to the image
    
    font = ImageFont.load_default()
    text = f"Predicted: {pred_class} | Truth: {truth_label}"
    text_color = (0, 255, 0) if pred_class == truth_label else (255, 0, 0)
    text_size = 16
    text_font = ImageFont.truetype("arial.ttf", text_size)
    text_width, text_height = draw.textsize(text, font=text_font)
    x = 10
    y = 10
    draw.text((x, y), text, font=text_font, fill=text_color)

    return new_img

def import_jpg_from_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over the files and find the first JPG file
    for file_name in file_list:
        if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
            # Construct the full path to the JPG file
            photo_path = os.path.join(folder_path, file_name)
            return photo_path
    
    return None

def delete_all_files_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
        print(f"File '{file_name}' deleted successfully.")

def extract_text_from_folder_names(folder_path):
    folder_names = os.listdir(folder_path)
    extracted_text = []
    for folder_name in folder_names:
        # Extract text from folder name (you can use any desired method)
        text = folder_name.split("_")[0]  # Split by underscore and take the first part
        
        extracted_text.append(text)
    
    return extracted_text


def delete_folders(folder_path):
    # Delete all folders within the specified directory
    for folder_name in os.listdir(folder_path):
        folder_to_delete = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_to_delete):
            shutil.rmtree(folder_to_delete)