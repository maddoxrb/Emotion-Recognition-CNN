import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

print("Initializing....")

train_transform = transforms.Compose([
    transforms.Resize(224),  # resize shortest side to 224 pixels
    transforms.CenterCrop(224),  # crop longest side to 224 pixels at center
    transforms.RandomRotation(10),  # rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # reverse 50% of images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=("C:\\Users\\Maddox\\OneDrive\\Desktop\\emotion-Recognition-CNN\\Emotion-Recognition-CNN\\FlowerDataset\\flowers"), transform=train_transform)

class_names = dataset.classes

train_indices, test_indices = train_test_split(list(range(len(dataset.targets))), test_size=0.2, stratify=dataset.targets)
train_data = torch.utils.data.Subset(dataset, train_indices)
test_data = torch.utils.data.Subset(dataset, test_indices)

train_data_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=10)

for images, labels in train_data_loader:
    break

im = make_grid(images, nrow=5)

inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
im = inv_normalize(im)

# Transfer learning with ResNet
class ResNetTransfer(nn.Module):
    def __init__(self, num_classes):
        super(ResNetTransfer, self).__init__()
        resnet = models.resnet18(weights=True)
        num_filters = resnet.fc.in_features
        resnet.fc = nn.Linear(num_filters, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

transfer_model = ResNetTransfer(7)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.001)

train_losses = []
test_losses = []
train_correct = []
test_correct = []
epochs = 10

print("Training Sequence Initiated")
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
     
    for batch_idx, (X_train, y_train) in enumerate(train_data_loader):
        batch_idx += 1

        transfer_model.train()
        
        y_pred = transfer_model(X_train)
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"epoch: {i} Batch: {batch_idx} Accuracy: {trn_corr.item()*100/(10*batch_idx):7.3f}%")
    
    loss = loss.detach().numpy()
    train_losses.append(loss)
    train_correct.append(trn_corr)
    
    transfer_model.eval()
    with torch.no_grad():
        for batch_idx, (X_test, y_test) in enumerate(test_data_loader):
            y_val = transfer_model(X_test)
            loss = criterion(y_val, y_test)
            
            predicted = torch.max(y_val.data, 1)[1]
            batch_corr = (predicted == y_test).sum()
            tst_corr += batch_corr
        
        loss = loss.detach().numpy()
        test_losses.append(loss)
        test_correct.append(tst_corr)
    
    

torch.save(transfer_model.state_dict(), 'Transfer_Model_Weights_FINAL_FINAL.pth')

device = torch.device("cpu")   #"cuda:0"

transfer_model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_data_loader:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = transfer_model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
print(y_pred[0:5])

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
