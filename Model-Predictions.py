import torch  
import random
import torch.nn as nn  
import torch.nn.functional as F
from torchvision import transforms, models  
from PIL import Image
from collections import Counter

def transfer_model_predict(image_path):
    #Create class definition
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                resnet = models.resnet18()  # Load the pre-trained ResNet-18 model
                num_filters = resnet.fc.in_features  # Get the number of input features for the fully connected layer
                resnet.fc = nn.Linear(num_filters, 7)  # Replace the fully connected layer with a new one for the specified number of classes
                self.resnet = resnet

            def forward(self, x):
                    return self.resnet(x)

        # Instantiate the model and load the trained weights
        model = CNN()
        model.load_state_dict(torch.load("Transfer-Model-Final-Weights.pth"))
        model.eval()

        # Define the data transformation for the single input image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension

        # Make the prediction
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        # Map the predicted class index to the corresponding label
        class_names = ['bellflower', 'daisy', 'dandelion', 'lotus', 'rose', 'sunflower', 'tulip']  # List of class labels
        predicted_label = class_names[predicted_class.item()]

        print(f'The predicted label is: {predicted_label}')

def custom_model_predict(image_path):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=nn.Conv2d(3,6,3,1)
            self.conv2=nn.Conv2d(6,16,3,1)
            self.fc1=nn.Linear(16*54*54,120) 
            self.fc2=nn.Linear(120,84)
            self.fc3=nn.Linear(84,20)
            self.fc4=nn.Linear(20,7)
        def forward(self,X):
            X=F.relu(self.conv1(X))
            X=F.max_pool2d(X,2,2)
            X=F.relu(self.conv2(X))
            X=F.max_pool2d(X,2,2)
            X=X.view(-1,16*54*54)
            X=F.relu(self.fc1(X))
            X=F.relu(self.fc2(X))
            X=F.relu(self.fc3(X))
            X=self.fc4(X)
            
            return F.log_softmax(X,dim=1)

    # Instantiate the model and load the trained weights
    model = CNN()
    model.load_state_dict(torch.load("Custom-Model-Final-Weights.pth"))
    model.eval()

    # Define the data transformation for the single input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the single image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    # Map the predicted class index to the corresponding label
    class_names = ['bellflower', 'daisy', 'dandelion', 'lotus', 'rose', 'sunflower', 'tulip']  # List of class labels
    predicted_label = class_names[predicted_class.item()]

    print(f'The predicted label is: {predicted_label}')

def ensemble_model_predict(image_path):
    class_names = ['bellflower', 'daisy', 'dandelion', 'lotus', 'rose', 'sunflower', 'tulip']  # List of class labels

    # Define the ModifiedConvolutionalNetwork class
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

    # Load the saved ensemble model weights
    ensemble_weights = torch.load('Ensemble-Model-Final-Weights.pth')

    # Create a list to store individual models
    ensemble_models = []

    # Create an instance of the ModifiedConvolutionalNetwork model for each weight
    for weight in ensemble_weights:
        model = ModifiedConvolutionalNetwork()
        model.load_state_dict(weight)
        model.eval()
        ensemble_models.append(model)

    # Load and preprocess the image
    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension

    # Make predictions with each model in the ensemble
    predictions = []

    for model in ensemble_models:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.exp(output)
            predicted_class = torch.argmax(probabilities, dim=1)
            predictions.append(predicted_class.item())

    # Combine predictions from all models using majority voting
    ensemble_prediction = Counter(predictions).most_common(1)[0][0]

    # Get the predicted class label
    predicted_label = class_names[ensemble_prediction]

    print("Prediction:", predicted_label)

    
      
names = ['bellflowers', 'daisys', 'dandelions', 'lotuss', 'roses', 'sunflowers', 'tulips']
random_image = random.choice(names) + str(random.randint(1,20)) +".jpg"

print(random_image)
transfer_model_predict(random_image)
custom_model_predict(random_image)
ensemble_model_predict(random_image)
