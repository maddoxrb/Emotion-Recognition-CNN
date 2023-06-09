import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from torchvision import models


print("Initializing....")

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = []
        self.labels = []
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Traverse through the subfolders to collect image paths and labels
        for label in os.listdir(data_dir):
            subfolder = os.path.join(data_dir, label)
            if os.path.isdir(subfolder):
                for image_file in os.listdir(subfolder):
                    self.image_files.append(os.path.join(subfolder, image_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = image.resize((48, 48))  # Resize to 48x48 pixels

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        label = self.classes.index(label)
        return image, label


# Specify the data transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Randomly rotate the image by up to 15 degrees
    transforms.GaussianBlur(kernel_size=3), # Apply Gaussian blur with a kernel size of 3
    transforms.ToTensor(),                   # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image pixel values
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define the paths to your train and test datasets
train_data_dir = 'C:\\Users\\Maddox\\Downloads\\archive (3)\\train'
test_data_dir = 'C:\\Users\\Maddox\\Downloads\\archive (3)\\test'

# Create instances of the train and test datasets
train_dataset = CustomDataset(data_dir=train_data_dir, transform=train_transform)
test_dataset = CustomDataset(data_dir=test_data_dir, transform=test_transform)

# Compute class weights
class_weights = []
total_samples = len(train_dataset)
num_classes = len(train_dataset.classes)
for label in train_dataset.classes:
    count = train_dataset.labels.count(label)
    weight = total_samples / (num_classes * count)
    class_weights.append(weight)

# Convert class weights to tensor
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Create data loaders
batch_size = 64
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the CNN model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Training loop
num_epochs = 25

for epoch in range(num_epochs):
    # Training
    train_loss = 0.0
    model.train()
    for images, labels in train_data_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)

    # Evaluation
    test_loss = 0.0
    test_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_data_loader:

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions = (predicted == labels).sum().item()
            test_accuracy += correct_predictions

        test_loss /= len(test_dataset)
        test_accuracy /= len(test_dataset)
        test_accuracy *= 100

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")