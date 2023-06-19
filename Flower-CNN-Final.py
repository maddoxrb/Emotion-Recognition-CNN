import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("Initializing....")

# Data preprocessing
train_transform = transforms.Compose([
    transforms.Resize(224),             # Resize the image to 224x224 pixels
    transforms.CenterCrop(224),         # Center crop the image to 224x224 pixels
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1, 1)),  # Apply Gaussian blur to the image
    transforms.RandomRotation(15),      # Randomly rotate the image by 15 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),              # Convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image
])

dataset = datasets.ImageFolder(root=("C:\\Users\\Maddox\\OneDrive\\Desktop\\emotion-Recognition-CNN\\Emotion-Recognition-CNN\\FlowerDataset\\flowers"), transform=train_transform)

train_indices, test_indices = train_test_split(list(range(len(dataset.targets))), test_size=0.2, stratify=dataset.targets)
train_data = torch.utils.data.Subset(dataset, train_indices)
test_data = torch.utils.data.Subset(dataset, test_indices)

train_data_loader = DataLoader(train_data, batch_size=10, shuffle=True)  # DataLoader for training data
test_data_loader = DataLoader(test_data, batch_size=10)  # DataLoader for test data

for images, labels in train_data_loader:
    break
batch_images = make_grid(images, nrow=5)  # Create a grid of batch images for visualization
inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])  # Inverse normalize the batch images for visualization
batch_images = inv_normalize(batch_images)
class_names = dataset.classes  # Get the class names from the dataset

# Define a modified convolutional network class
class ModifiedConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)  # First convolutional layer: input channels=3, output channels=6, kernel size=3x3, stride=1
        self.conv2 = nn.Conv2d(6, 16, 3, 1)  # Second convolutional layer: input channels=6, output channels=16, kernel size=3x3, stride=1
        self.fc1 = nn.Linear(46656, 120)  # First fully connected layer: input features=46656, output features=120
        self.fc2 = nn.Linear(120, 84)  # Second fully connected layer: input features=120, output features=84
        self.fc3 = nn.Linear(84, 20)  # Third fully connected layer: input features=84, output features=20
        self.fc4 = nn.Linear(20, 7)  # Fourth fully connected layer: input features=20, output features=7
    
    def forward(self, X):
        X = self.conv1(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2, 2)  # Apply max pooling to downsample the feature maps
        X = self.conv2(X)
        X = F.relu(X)
        X = F.max_pool2d(X, 2, 2)  
        X = X.view(-1, 16*54*54)  # Reshape the feature maps into a 1D vector
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)  # Output layer
        X = F.log_softmax(X, dim=1)

        return X

modified_model = ModifiedConvolutionalNetwork()  # Create an instance of the modified convolutional network
criterion = nn.CrossEntropyLoss()  # Define the loss function
optimizer = torch.optim.Adam(modified_model.parameters(), lr=0.001)  # Define the optimizer with a learning rate of 0.001

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)  # Reduce learning rate when the validation loss plateaus

train_correct = []  # List to store the number of correct predictions during training
test_correct = []  # List to store the number of correct predictions during testing
train_losses = []  # List to store the training loss at each epoch
test_losses = []  # List to store the testing loss at each epoch
epochs = 2  # Number of epochs for training

print("Training Sequence Initiated")

# Training loop
for i in range(epochs):
    trn_corr = 0  # Counter for the number of correct predictions during training
    tst_corr = 0  # Counter for the number of correct predictions during testing
    
    for batch_idx, (X_train, y_train) in enumerate(train_data_loader):
        batch_idx += 1

        modified_model.train()  # Set the model to training mode
        
        y_pred = modified_model(X_train)  # Forward pass
        loss = criterion(y_pred, y_train)  # Calculate the loss
        predicted = torch.max(y_pred.data, 1)[1]  # Get the predicted labels
        batch_corr = (predicted == y_train).sum()  # Count the number of correct predictions in the batch
        trn_corr += batch_corr  # Accumulate the total number of correct predictions

        optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        if batch_idx % 200 == 0:
            print(f"epoch: {i} Batch: {batch_idx} Accuracy: {trn_corr.item()*100/(10*batch_idx):7.3f}%")
    
    loss = loss.detach().numpy()  # Detach the loss from the computation graph and convert it to a numpy array
    train_losses.append(loss)  # Store the training loss for the current epoch
    train_correct.append(trn_corr)  # Store the number of correct predictions for the current epoch
    
    modified_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch_idx, (X_test, y_test) in enumerate(test_data_loader):
            y_val = modified_model(X_test)  # Forward pass on the test data
            loss = criterion(y_val, y_test)  # Calculate the loss
            
            predicted = torch.max(y_val.data, 1)[1]  # Get the predicted labels
            batch_corr = (predicted == y_test).sum()  # Count the number of correct predictions in the batch
            tst_corr += batch_corr  # Accumulate the total number of correct predictions

        loss = loss.detach().numpy()  # Detach the loss from the computation graph and convert it to a numpy array
        test_losses.append(loss)  # Store the testing loss for the current epoch
        test_correct.append(tst_corr)  # Store the number of correct predictions for the current epoch
        
    scheduler.step(loss)  # Update the learning rate based on the validation loss

torch.save(modified_model.state_dict(), 'F_modified.pth')  # Save the trained model parameters to a file

device = torch.device("cpu")   #"cuda:0"  # Choose the device for inference (CPU in this case)

modified_model.eval()  # Set the model to evaluation mode
y_true = []  # List to store the true labels
y_pred = []  # List to store the predicted labels
with torch.no_grad():
    for test_data in test_data_loader:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)  # Move the test data to the chosen device
        pred = modified_model(test_images).argmax(dim=1)  # Forward pass on the test data and get the predicted labels
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())  # Append the true label to the list
            y_pred.append(pred[i].item())  # Append the predicted label to the list
print(y_pred[0:5])  # Print the predicted labels for the first 5 test samples

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))  # Print the classification report with precision, recall, and F1-score for each class