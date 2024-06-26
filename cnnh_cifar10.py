import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init
import sys
sys.path.insert(0, sys.path[0]+"/../")
from utill import data_load,getsimilarity

class CNNH(nn.Module):
    def __init__(self,numbers_bit):
        super().__init__()
        self.conv=nn.Sequential(
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
        self.fc=nn.Sequential(
            nn.Linear(128 * 3 * 3, 500),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(500, numbers_bit),
        )
        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def hash_loss(output,h,batch_size):
    """
    calculate the loss
    :param output: the output of the model
    :param h: the approximate hash
    :return: loss
    """
    h=torch.as_tensor(h)
    loss_1=torch.sqrt(torch.sum(torch.pow(output-h,2)))/batch_size
    return loss_1
    pass

def train_and_test(batch_size, hash_bit, num_epochs):
    train_losses = []
    test_losses = []

    model = CNNH(hash_bit)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    data_train_load, data_test_load = data_load('C:/Users/rocky/OneDrive/learndata/machine learning/final_work/CNNH/CNNH/data/cifar', batch_size)

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

train_and_test(1000, 10, 4)

 
