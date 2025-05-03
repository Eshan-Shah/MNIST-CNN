from torchvision import datasets
from torch.utils.data import DataLoader

def load_data(batch_size=64):
    # Load raw MNIST dataset without any transforms
    train_dataset = datasets.MNIST(root='./MNIST-DATA', train=True, download=True)
    test_dataset = datasets.MNIST(root='./MNIST-DATA', train=False, download=True)

    # Wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
