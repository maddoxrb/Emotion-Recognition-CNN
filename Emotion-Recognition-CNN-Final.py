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
        if label == 'angry':
            label = 0
        if label == 'disgust':
            label = 1
        if label == 'fear':
            label = 2
        if label == 'happy':
            label = 3
        if label == 'neutral':
            label = 4
        if label == 'sad':
            label = 5
        if label == 'surprise':
            label = 6
        return image, label

# Specify the data transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Randomly rotate the image by up to 15 degrees
    #transforms.RandomCrop(size=224),        # Randomly crop the image
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

# Create data loaders
batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # First fully connected layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer after the first fully connected layer
        self.fc2 = nn.Linear(256, 128)  # Second fully connected layer
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer after the second fully connected layer
        self.fc3 = nn.Linear(128, 64)  # Third fully connected layer
        self.dropout3 = nn.Dropout(0.5)  # Dropout layer after the third fully connected layer
        self.fc4 = nn.Linear(64, 7)  # Output size is 7 for the number of emotions

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)  # Apply dropout after the first fully connected layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)  # Apply dropout after the second fully connected layer
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)  # Apply dropout after the third fully connected layer
        x = self.fc4(x)
        return x

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    print("Processing Epoch " + str(epoch) + ": ....")
    model.train()
    train_loss = 0.0

    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_dataset)

    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_dataset)
    test_accuracy = correct / len(test_dataset) * 100

    scheduler.step()  # Update learning rate scheduler

    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

torch.save(model.state_dict(), 'emotion_classifier.pth')