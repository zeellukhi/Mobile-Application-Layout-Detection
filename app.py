import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Define the directory where the dataset and classes.txt are located
dataset_dir = './new_dataset/data/'

# Load class mapping from classes.txt
class_mapping = {}
classes_file_path = os.path.join(dataset_dir, 'classes.txt')



with open(classes_file_path, 'r') as class_file:
    classes = class_file.read().strip().split('\n')
    for i, class_name in enumerate(classes):
        class_mapping[i] = class_name

# Define the number of classes (layout components)
num_classes = len(class_mapping)

# Initialize the model
model_checkpoint_path = 'efficientnet_layout_detection.pth'  # Update with your model checkpoint path
efficientnet_variant = 'efficientnet-b0'
model = EfficientNet.from_pretrained(efficientnet_variant, num_classes=num_classes)
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()

# Initialize predictions as an empty list
predictions = []
labels = []

def preprocess_image(image_path):
    # Implement image preprocessing here
    # Remember to resize the image to 224x224 pixels and normalize it
    # You can use the transforms from PyTorch or create your own
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Fixed size, adjust as needed
        transforms.CenterCrop((224, 224)),  # Crop to maintain the aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use appropriate mean and std values
    ])
    
    image = Image.open(image_path)
    image = transform(image)
    return image

def get_preds(image_path):
    global predictions  # Use the global predictions list

    # Clear previous predictions
    predictions = []

    # Preprocess the image
    image = preprocess_image(image_path)

    outputs = model(image.unsqueeze(0))  # Assuming your model expects a batch of images

    # Convert predictions to binary (0 or 1)
    binary_predictions = (torch.sigmoid(outputs) > 0.5).float()

    # Iterate through batch samples
    for i in range(binary_predictions.shape[0]):
        prediction = binary_predictions[i]
        image_height, image_width = image.size(1), image.size(2)

        # Iterate through the predicted classes
        for class_id in range(num_classes):
            if prediction[class_id] == 1:
                # Calculate bounding box coordinates and confidence
                confidence = prediction[class_id].item()
                x_center, y_center = (i + 0.5) * (image_width / 224), (i + 0.5) * (image_height / 224)
                width, height = image_width / 224, image_height / 224

                # Append the YOLO format prediction to the list
                predictions.append([class_id, x_center, y_center, width, height])

def getlabel(image_path):
    # Define the image transformation
    test_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_image_tensor = transform(test_image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(test_image_tensor)
        predicted_labels = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()

    predicted_labels = [class_mapping[i] for i, pred in enumerate(predicted_labels) if i in class_mapping and pred == 1]

    return predicted_labels



@app.route('/')
def index():
    return render_template('index.html', predictions=[])

@app.route('/predict', methods=['POST'])
def predict():
    global predictions  # Use the global predictions list
    global labels  # Use the global predictions list

    # Check if an image file is included in the request
    if 'image' not in request.files:
        return render_template('index.html', predictions=[ ])

    image = request.files['image']

    # Make predictions on the uploaded image
    get_preds(image)
    labels = getlabel(image)
    # Render the HTML template with the predictions
    return render_template('index.html', predictions=predictions, labels=labels)

if __name__ == '_main_':
    app.run(debug=True)