import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from stage1.lahc import learning_hash
import torch
def data_load(path,batch_size):
    """load dataset"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #be careful
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_loader = DataLoader(CIFAR10(path, train=True, download=True, transform=transform_train),
                              batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print('train set: {}'.format(len(train_loader.dataset)))
    test_loader = DataLoader(CIFAR10(path, train=False, download=True, transform=transform_test),
                             batch_size=batch_size * 8, shuffle=False, num_workers=0, pin_memory=True)
    print('val set: {}'.format(len(test_loader.dataset)))

    # data_train_iter=iter(train_loader)
    # data_test_iter=iter(test_loader)

    # image_train,label_train=data_train_iter.next()
    # image_test, label_test = data_test_iter.next()

    return train_loader,test_loader
    #pass

def data_load2(annFile_train, root_train, annFile_val, root_val, batch_size):
    """Load dataset"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),  # Adjust image size
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust image size
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = torchvision.datasets.CocoCaptions(root=root_train, annFile=annFile_train, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    print('train set: {}'.format(len(train_loader.dataset)))
    
    test_dataset = torchvision.datasets.CocoCaptions(root=root_val, annFile=annFile_val, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 8, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    print('val set: {}'.format(len(test_loader.dataset)))

    return train_loader, test_loader

def collate_fn(batch):
    """Custom collate function to handle images and captions"""
    images, captions = zip(*batch)
    images = [transforms.Resize((256, 256))(image) for image in images]  # Ensure all images are resized
    images = torch.stack(images, 0)
    # Converting captions to indices or some other numeric representation
    captions = [caption[0] for caption in captions]  # Assuming each caption is a list with one string element
    return images, captions



def getsimilarity(labels, numbers_bit):
    """
    Calculate the similarity based on the labels.
    :param labels: list of labels
    :param numbers_bit: number of bits for the hash
    :return: similarity matrix
    """
    # Convert labels to numpy array if needed
    if isinstance(labels, (list, tuple)):
        labels = np.array([label for label in labels])
    else:
        labels = np.array(labels)

    # Example similarity calculation, replace with actual logic
    similarity_matrix = np.zeros((len(labels), numbers_bit))
    for i, label in enumerate(labels):
        similarity_matrix[i, :] = np.random.rand(numbers_bit)  # Replace with actual similarity calculation
    
    return torch.tensor(similarity_matrix, dtype=torch.float)
# data_train_iter,data_test_iter = data_load('/Users/Dream/PycharmProjects/CNNH/data/cifar', 1001)
# for i in range(50):
#     image_train,label_train=next(data_train_iter)
#     print(len(image_train))
# image_train,image_test,label_train,label_test=dataLoad('/Users/Dream/PycharmProjects/CNNH/data/cifar',64)
# similarity=getSimilarity(label_train)
# h=learning_hash(similarity,32,10,0.2)