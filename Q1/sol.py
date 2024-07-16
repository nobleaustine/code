import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch


def one_hot_encode(labels, num_classes=10):
    
    one_hot = torch.zeros((labels.size(0), num_classes))
    rows = torch.arange(labels.size(0))
    one_hot[rows, labels] = 1
    return one_hot

def visualize_image(image, label ):

    if torch.is_tensor(image):
        image = image.squeeze().numpy()  

    plt.imshow(image, cmap='gray')  
    plt.title(f'Label: {label}')
    plt.axis('off')  
    plt.show()

class MINSTDataset(Dataset):

    def __init__(self, data, flag="train",transform=None):

        
        print("preprocessing data...")
        
        self.data = data.data.float().unsqueeze(1) 
        self.targets = one_hot_encode(data.targets) # c one hot encoding the label
        self.transform = transform

        # b normalizing the image
        if self.transform:
            self.data = self.transform(self.data)

        print("splitting data...")
        # d splitting the data
        s1 = int(len(data)*0.8)
        s2 = int(len(data)*0.9)

        if flag == "train":
            self.data = self.data[:s1]
            self.targets = self.targets[:s1] 
        elif flag == "test":
            self.data = self.data[s1:s2]
            self.targets = self.targets[s1:s2]
        elif flag == "val":
            self.data = self.data[s2:]
            self.targets = self.targets[s2:] 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]
        label = self.targets[idx]

        return image, label

def main(path = '../downloads/data'):

    # a. Load the MNIST handwritten digit dataset (e)
    data = datasets.MNIST(root=path, train=True, download=True)
    transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])

    dataset = MINSTDataset(data, flag="train", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for image, label in loader:

        visualize_image(image[0], label[0])
        break

if __name__ == "__main__":
    main()