# app.py
import io
import os
import uuid
import random
import torch
from flask import Flask, render_template, send_file, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from src.model_load import  modelV3
from src.updated_model_load import updated_model
from src.data_load import test_data, names, le
import numpy as np
from src.app_functions import make_predictions, random_image, preprocess_image, draw_picture, import_jpg_from_folder,delete_all_files_in_folder, extract_text_from_folder_names, delete_folders

app = Flask(__name__)

loaded_model = modelV3

load_updated_model = updated_model


loaded_test_data = test_data

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload_images', methods=['POST'])
def upload_images():
    delete_folders("uploads")

    if os.path.exists("updated_model.pth"):
        os.remove("updated_model.pth")
        print("Model file deleted successfully.")
    else:
        print("Model file not found.")
    # Get the text input from the user
    folder_name = request.form['folder_name']

    # Create a folder with the provided text
    folder_path = os.path.join('uploads', folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Get the uploaded images
    images = request.files.getlist('images')

    # Save each image to the folder
    for image in images:
        image.save(os.path.join(folder_path, image.filename))

    

    return "images upload"

@app.route('/upload_image_for_testing', methods=['POST'])
def upload_image_for_testing():
    delete_all_files_in_folder("tmp")
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    
    if image.filename == '':
        return 'No image selected', 400

    try:
        filename = image.filename
        image.save(os.path.join('tmp', filename))
        return 'Image uploaded successfully'
    except Exception as e:
        return f'Error uploading image: {str(e)}', 500

@app.route('/testing_image')
def load_test_image():
    image_path = import_jpg_from_folder("tmp")
    image = Image.open(image_path)
    preproced_image, label = preprocess_image(image)
    print("preproced done")
    pred_probs = make_predictions(model=updated_model,
                              data=[preproced_image])
    
    
    pred_classes = pred_probs.argmax(dim=1)
    print("predict done")

    if pred_classes < 159:
        pred_class = names[le.inverse_transform([pred_classes])[0]]
    else:

        pred_class = extract_text_from_folder_names('uploads')[0]
    # Get the ground truth label
    #  
    truth_label = extract_text_from_folder_names('uploads')[0]
    # Get picture
    new_imge = draw_picture(preproced_image, pred_class, truth_label)


    buffer = io.BytesIO()
    new_imge.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

@app.route('/random_image')
def get_random_image():
    img, label = random_image(loaded_test_data)

    # Load the model
    model = loaded_model

    # Make predictions on test samples with model 
    
    pred_probs = make_predictions(model=model,
                              data=[img])
    
    pred_classes = pred_probs.argmax(dim=1)
    # Get the predicted class name
    pred_class = names[le.inverse_transform([pred_classes])[0]]

    # Get the ground truth label
    truth_label = names[le.inverse_transform([label])[0]] 

    # Get picture
    new_img = draw_picture(img, pred_class, truth_label)

    buffer = io.BytesIO()
    new_img.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)