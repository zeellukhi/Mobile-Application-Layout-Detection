import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import wandb


wandb.init(project='layout_detection', entity='zeellukhi99', config={})

# Define paths and directories
dataset_dir = './new_dataset/data/'  # Replace with the actual path
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')  # Path to your validation set

# Create a class mapping from classes.txt
class_mapping = {}
classes_file_path = os.path.join(dataset_dir, 'classes.txt')

with open(classes_file_path, 'r') as class_file:
    classes = class_file.read().strip().split('\n')
    for i, class_name in enumerate(classes):
        class_mapping[i] = class_name

# Define the number of classes (layout components)
num_classes = len(class_mapping)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Fixed size, adjust as needed
    transforms.CenterCrop((224, 224)),  # Crop to maintain the aspect ratio
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use appropriate mean and std values
])

class CustomLayoutDataset(Dataset):
    def __init__(self, root_dir, num_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith(".png") or filename.endswith(".jpg")]
        self.annotation_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith(".txt")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure images have 3 channels (RGB)
        labels = self._parse_annotation(annotation_path)

        if self.transform:
            image = self.transform(image)

        # Create a multi-label target tensor
        target = torch.zeros(self.num_classes)
        target[labels] = 1

        return image, target

    def _parse_annotation(self, annotation_path):
        with open(annotation_path, 'r') as annotation_file:
            lines = annotation_file.read().strip().split('\n')
        annotations = [line.split() for line in lines]
        labels = [int(annotation[0]) for annotation in annotations]
        return labels
    

# Define the number of classes (layout components)
num_classes = len(class_mapping)

# Modify your dataset creation to pass in the number of classes
train_dataset = CustomLayoutDataset(train_dir, num_classes, transform=transform)


# Create a DataLoader with a custom collate function
def custom_collate(batch):
    images, labels_list = zip(*batch)
    labels_lengths = [len(labels) for labels in labels_list]
    max_labels_length = max(labels_lengths)
    
    # Pad labels with -1 (or any other suitable value) to create a tensor of variable length labels
    padded_labels = torch.zeros(len(batch), max_labels_length).long() - 1
    for i, labels in enumerate(labels_list):
        padded_labels[i, :len(labels)] = torch.tensor(labels)
    
    return torch.stack(images), padded_labels, torch.tensor(labels_lengths)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

val_dataset = CustomLayoutDataset(val_dir, num_classes, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

def calculate_accuracy(predictions, targets):
    predictions = torch.sigmoid(predictions)
    binary_predictions = (predictions > 0.5).float()
    accuracy = accuracy_score(targets.cpu().numpy(), binary_predictions.cpu().numpy())
    return accuracy


# # Load the EfficientNet pretrained model
# efficientnet_variant = 'efficientnet-b0'  # Choose the variant you want
# efficientnet_base = EfficientNet.from_pretrained(efficientnet_variant, num_classes=num_classes)

# Step 1: Load the pre-trained EfficientNet-B0 model
efficientnet_variant = 'efficientnet-b0'
efficientnet_base = EfficientNet.from_pretrained(efficientnet_variant)

# Freeze the pretrained layers
for param in efficientnet_base.parameters():
    param.requires_grad = False

# Step 2: Replace global average pooling with adaptive average pooling
efficientnet_base._avg_pooling = nn.AdaptiveAvgPool2d(1)  # Output will be 1x1

# Step 3: Add a dropout layer after the adaptive average pooling
dropout_rate = 0.2
efficientnet_base._dropout = nn.Dropout(p=dropout_rate)

# Step 4: Retrieve the number of input features to the final fully connected layer
num_ftrs = efficientnet_base._fc.in_features

# Step 5: Replace the final fully connected layer with a new linear layer
efficientnet_base._fc = nn.Linear(num_ftrs, num_classes)

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(efficientnet_base.parameters(), lr=0.001)

# Training loop
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    efficientnet_base.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in train_loader:
        images, labels, labels_lengths = batch  # Unpack the batch
        optimizer.zero_grad()

        # Forward pass
        outputs = efficientnet_base(images)

        loss = criterion(outputs, labels.float())  # Use BCEWithLogitsLoss and convert labels to float

        # Calculate accuracy
        correct_predictions += (torch.sum((outputs > 0.5).float() == labels)).item()
        total_samples += labels.numel()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate training accuracy
    training_accuracy = correct_predictions / total_samples

    # Validation loop
    efficientnet_base.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            images, labels, labels_lengths = batch
            outputs = efficientnet_base(images)
            val_loss += criterion(outputs, labels.float()).item()
            val_correct_predictions += (torch.sum((outputs > 0.5).float() == labels)).item()
            val_total_samples += labels.numel()

    # Calculate validation accuracy
    validation_accuracy = val_correct_predictions / val_total_samples
    avg_val_loss = val_loss / len(val_loader)
    training_loss=running_loss / len(train_loader)
    # Print metrics for this epoch
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Training Loss: {training_loss}')
    print(f'Training Accuracy: {training_accuracy * 100:.2f}%')
    print(f'Validation Loss: {avg_val_loss}')
    print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')
    # Log metrics to wandb
    wandb.log({"Training Loss": training_loss, "Training Accuracy": training_accuracy, "Validation Loss": avg_val_loss, "Validation Accuracy": validation_accuracy})

    
torch.save(efficientnet_base.state_dict(), 'efficientnet_layout_detection.pth')

# Define the path to the test dataset folder
test_dir = os.path.join(dataset_dir, 'test')

# Create a CustomLayoutDataset for the test dataset
test_dataset = CustomLayoutDataset(test_dir, num_classes, transform=transform)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

# Set the model to evaluation mode
efficientnet_base.eval()

# Initialize variables to calculate metrics
test_correct_predictions = 0
test_total_samples = 0
test_predictions = []
test_targets = []

# Iterate through the test DataLoader
with torch.no_grad():
    for batch in test_loader:
        images, labels, labels_lengths = batch
        outputs = efficientnet_base(images)

        # Convert predictions to binary (0 or 1)
        binary_predictions = (torch.sigmoid(outputs) > 0.5).float()
        
        # Update predictions and targets for evaluation
        test_predictions.append(binary_predictions.cpu().numpy())
        test_targets.append(labels.cpu().numpy())

        # Calculate the number of correct predictions
        test_correct_predictions += (binary_predictions == labels).sum().item()
        test_total_samples += labels.numel()

# Calculate accuracy on the test dataset
test_accuracy = test_correct_predictions / test_total_samples

# Flatten the lists of predictions and targets
test_predictions = np.vstack(test_predictions)
test_targets = np.vstack(test_targets)

# Calculate other evaluation metrics (e.g., precision, recall, F1-score)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

precision = precision_score(test_targets, test_predictions, average='micro')
recall = recall_score(test_targets, test_predictions, average='micro')
f1 = f1_score(test_targets, test_predictions, average='micro')

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Precision: {precision * 100:.2f}%')
print(f'Test Recall: {recall * 100:.2f}%')
print(f'Test F1-Score: {f1 * 100:.2f}%')

wandb.log({"Test Accuracy": test_accuracy, "Test Precision": precision, "Test Recall": recall, "Test F1-Score": f1})


wandb.watch(efficientnet_base)
wandb.finish()
