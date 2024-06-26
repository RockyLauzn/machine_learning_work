import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm
import sys
sys.path.insert(0, sys.path[0]+"/../")
from utill import data_load1, data_load2, getsimilarity

class CNNH(nn.Module):
    def __init__(self, numbers_bit):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 31 * 31, 500),  # Adjusted input size
            nn.ReLU(inplace=True),
            nn.Linear(500, numbers_bit),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def hash_loss(output, h, batch_size):
    """
    Calculate the loss.
    :param output: the output of the model
    :param h: the approximate hash
    :return: loss
    """
    h = torch.as_tensor(h)
    loss_1 = torch.sqrt(torch.sum(torch.pow(output - h, 2))) / batch_size
    return loss_1



def train_and_test1(batch_size, hash_bit, num_epochs):
    train_losses = []
    test_losses = []

    model = CNNH(hash_bit)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    data_train_load, data_test_load = data_load1('C:/Users/rocky/OneDrive/learndata/machine learning/bigwork/CNNH/CNNH/data/cifar', batch_size)

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        for i, (imgs, labels) in enumerate(data_train_load):
            H = getsimilarity(labels, numbers_bit=hash_bit)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = hash_loss(outputs, H, batch_size)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(data_train_load))

        epoch_test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, labels in data_test_load:
                H = getsimilarity(labels, numbers_bit=hash_bit)
                outputs = model(imgs)
                loss = hash_loss(outputs, H, batch_size)
                epoch_test_loss += loss.item()
        test_losses.append(epoch_test_loss / len(data_test_load))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")

    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.show()
    
    

    
def train_and_test2(batch_size, hash_bit, num_epochs):
    train_losses = []
    test_losses = []

    model = CNNH(hash_bit)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    data_train_load, data_test_load = data_load2(
        'C:/Users/rocky/OneDrive/learndata/machine learning/bigwork/CNNH/CNNH/data/coco/annotations/captions_train2017.json', 
        'C:/Users/rocky/OneDrive/learndata/machine learning/bigwork/CNNH/CNNH/data/coco/train2017/', 
        'C:/Users/rocky/OneDrive/learndata/machine learning/bigwork/CNNH/CNNH/data/coco/annotations/captions_val2017.json', 
        'C:/Users/rocky/OneDrive/learndata/machine learning/bigwork/CNNH/CNNH/data/coco/val2017/', 
        batch_size
    )

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        model.train()
        for i, (imgs, labels) in tqdm(enumerate(data_train_load), total=len(data_train_load), desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            # Assuming labels is a list of captions, process it accordingly
            H = getsimilarity(labels, numbers_bit=hash_bit)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = hash_loss(outputs, H, batch_size)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(data_train_load))

        epoch_test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(data_test_load, total=len(data_test_load), desc=f"Testing Epoch {epoch+1}/{num_epochs}"):
                H = getsimilarity(labels, numbers_bit=hash_bit)
                outputs = model(imgs)
                loss = hash_loss(outputs, H, batch_size)
                epoch_test_loss += loss.item()
        test_losses.append(epoch_test_loss / len(data_test_load))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")

    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.show()
# Training CIFAR dataset
#train_and_test1(1000, 10, 4)

# Training COCO dataset
train_and_test2(1000, 10, 2)