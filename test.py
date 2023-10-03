import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np

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

model_checkpoint_path = 'efficientnet_layout_detection.pth'  # Update with your model checkpoint path
efficientnet_variant = 'efficientnet-b0'
model = EfficientNet.from_pretrained(efficientnet_variant, num_classes=num_classes)
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()

def get_preds(image_path, model_path):
    yolo_predictions = []
    # efficientnet_variant = 'efficientnet-b0'
    # model = EfficientNet.from_pretrained(efficientnet_variant, num_classes=num_classes)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()

    # Preprocess the image
    image = preprocess_image(image_path)

    outputs = model(image.unsqueeze(0))  # Assuming your model expects a batch of images

    # Convert predictions to binary (0 or 1)
    binary_predictions = (torch.sigmoid(outputs) > 0.5).float()

    # Iterate through batch samples
    for i in range(binary_predictions.shape[0]):
                prediction = binary_predictions[i]
                image_height, image_width = image.size(1), image.size(2)
                # image_width, image_height = image.size(3), image.size(2)

                # Iterate through the predicted classes
                for class_id in range(num_classes):
                    if prediction[class_id] == 1:
                        # Calculate bounding box coordinates and confidence
                        confidence = prediction[class_id].item()
                        x_center, y_center = (i + 0.5) * (image_width / 224), (i + 0.5) * (image_height / 224)
                        width, height = image_width / 224, image_height / 224

                        # Append the YOLO format prediction to the list
                        yolo_predictions.append([class_id, x_center, y_center, width, height])
    return yolo_predictions

def draw_boxes_on_image(image_path, predictions, output_image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    for prediction in predictions:
        class_id, x_center, y_center, width, height = prediction
        
        # Convert YOLO format to coordinates
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        
        # Define the color and thickness of the bounding box
        color = (0, 255, 0)  # Green color in BGR
        thickness = 2
        
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # You can also add class labels if needed
        class_name = class_mapping[class_id]
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Save the image with bounding boxes
    cv2.imwrite(output_image_path, image)

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

      

# Specify the path to your image and model checkpoint
image_path = "./new_dataset/data/images/b5.jpg"  # Update with your image path
model_checkpoint_path = 'efficientnet_layout_detection.pth'  # Update with your model checkpoint path
output_image_path = './Annoted_Img/b5.jpg'

# Make predictions on the specified image
predictions = get_preds(image_path, model_checkpoint_path)

# Process model output as needed and display the predictions
# For example, you can print the predicted classes or use them for further processing.
labels = getlabel(image_path)
print(labels)  
print(predictions)

# Draw bounding boxes on the image and save it
draw_boxes_on_image(image_path, predictions, output_image_path)

print(f"Image with bounding boxes saved to {output_image_path}")
