import torch  
import torch.nn as nn  
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report  
from torchvision import datasets, transforms, models  
from torchvision.utils import make_grid  
from PIL import Image

# Define the CNN model architecture (same as before)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        resnet = models.resnet18(weights=True)  # Load the pre-trained ResNet-18 model
        num_filters = resnet.fc.in_features  # Get the number of input features for the fully connected layer
        resnet.fc = nn.Linear(num_filters, num_classes)  # Replace the fully connected layer with a new one for the specified number of classes
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

def classify(image_path):
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

photoList = []

for x in photoList:
    classify(x)