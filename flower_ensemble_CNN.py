import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import DataLoader  
from torchvision import datasets, transforms  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report  
from torch.optim.lr_scheduler import ReduceLROnPlateau  

print("Initializing....")  

# Data preprocessing
train_transform = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.CenterCrop(224),  # Crop the center of the image to 224x224
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1, 1)),  # Apply Gaussian blur to the image
    transforms.RandomRotation(15),  # Randomly rotate the image by 15 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
])

# Load the dataset
dataset = datasets.ImageFolder(root="C:\\Users\\Maddox\\OneDrive\\Desktop\\emotion-Recognition-CNN\\Emotion-Recognition-CNN\\FlowerDataset\\flowers", transform=train_transform)

class_names = dataset.classes  # Get the list of class names from the dataset

# Split the dataset into train and test sets
train_indices, test_indices = train_test_split(list(range(len(dataset.targets))), test_size=0.2, stratify=dataset.targets)
train_data = torch.utils.data.Subset(dataset, train_indices)
test_data = torch.utils.data.Subset(dataset, test_indices)

# Create data loaders for training and testing
train_data_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=10)

# Define the modified convolutional network model
class ModifiedConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(6, 16, 3, 1)  # Second convolutional layer
        self.fc1 = nn.Linear(16*54*54, 120)  # First fully connected layer
        self.fc2 = nn.Linear(120, 84)  # Second fully connected layer
        self.fc3 = nn.Linear(84, 20)  # Third fully connected layer
        self.fc4 = nn.Linear(20, len(class_names))  # Output layer
    
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

ensemble_models = []  # List to store the ensemble of models
num_models = 3  # Number of models in the ensemble

# Create and add models to the ensemble
for x in range(num_models):
    model = ModifiedConvolutionalNetwork()  # Create a new instance of the modified convolutional network
    ensemble_models.append(model)  # Add the model to the ensemble

criterion = nn.CrossEntropyLoss()  # Define the loss function
optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in ensemble_models]  # Create optimizers for each model

# Learning rate scheduler
schedulers = [ReduceLROnPlateau(optimizer, mode='min', patience=3) for optimizer in optimizers]

train_losses = []  # List to store the training losses
test_losses = []  # List to store the test losses
train_correct = []  # List to store the number of correct predictions in the training set
test_correct = []  # List to store the number of correct predictions in the test set
epochs = 20  # Number of training epochs

print("Training Sequence Initiated")  # Print a message indicating the start of the training sequence

# Training loop
for i in range(epochs):
    trn_corr = [0] * num_models  # Counter for correct predictions in the training set for each model
    tst_corr = [0] * num_models  # Counter for correct predictions in the test set for each model
    
    # Iterate over batches in the training set
    for batch_idx, (X_train, y_train) in enumerate(train_data_loader):
        for j, model in enumerate(ensemble_models):
            batch_idx += 1  # Increment batch index
            
            model.train()  # Set the model to training mode
            optimizer = optimizers[j]  # Get the optimizer for the current model

            y_pred = model(X_train)  # Forward pass
            loss = criterion(y_pred, y_train)  # Compute the loss
            predicted = torch.max(y_pred.data, 1)[1]  # Get the predicted labels
            batch_corr = (predicted == y_train).sum()  # Count correct predictions
            trn_corr[j] += batch_corr  # Update the counter for correct predictions

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model's parameters
            if batch_idx % 200 == 0:
                print(f"epoch: {i} Batch: {batch_idx} Accuracy: {trn_corr[j].item()*100/(10*batch_idx):7.3f}%")
    
    train_losses.append(loss.detach().numpy())  # Append the loss value to the train_losses list
    train_correct.append(trn_corr)  # Append the correct predictions counter to the train_correct list
    
    # Evaluate the models on the test set
    for j, model in enumerate(ensemble_models):
        model.eval()  # Set the model to evaluation mode
        tst_corr[j] = 0  # Reset the counter for correct predictions

        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, (X_test, y_test) in enumerate(test_data_loader):
                y_val = model(X_test)  # Forward pass
                loss = criterion(y_val, y_test)  # Compute the loss
                predicted = torch.max(y_val.data, 1)[1]  # Get the predicted labels
                batch_corr = (predicted == y_test).sum()  # Count correct predictions
                tst_corr[j] += batch_corr  # Update the counter for correct predictions
        
        test_losses.append(loss.detach().numpy())  # Append the loss value to the test_losses list
        test_correct.append(tst_corr)  # Append the correct predictions counter to the test_correct list
        accuracy = tst_corr[j].item() / len(test_data_loader.dataset) * 100.0  # Compute the accuracy
        print(f"epoch: {i} Model: {j+1} Testing Accuracy: {accuracy:7.3f}%")
        
        schedulers[j].step(loss)  # Update the learning rate based on the validation loss

# Combine predictions from all models in the ensemble
device = torch.device("cpu")  # Use the CPU device for inference
y_true = []  # List to store the true labels
y_pred_ensemble = []  # List to store the ensemble predictions

# Iterate over batches in the test set
for test_data in test_data_loader:
    test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
    predictions = []
    
    for model in ensemble_models:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            pred = model(test_images).argmax(dim=1)  # Get the predicted labels
            predictions.append(pred)  # Add the predictions to the list
    
    # Majority voting for ensemble predictions
    predictions = torch.stack(predictions)
    pred_ensemble = predictions.mode(dim=0).values  # Get the most frequent predictions
    
    for i in range(len(pred_ensemble)):
        y_true.append(test_labels[i].item())  # Add the true label to the list
        y_pred_ensemble.append(pred_ensemble[i].item())  # Add the ensemble prediction to the list

print(y_pred_ensemble[0:5])  # Print the first 5 predictions

print(classification_report(y_true, y_pred_ensemble, target_names=class_names, digits=4))  # Generate a classification report

ensemble_weights = [model.state_dict() for model in ensemble_models]  # Get the state dictionaries of the ensemble models
torch.save(ensemble_weights, 'ensemble_model.pth')  # Save the combined weights of the ensemble model to a file