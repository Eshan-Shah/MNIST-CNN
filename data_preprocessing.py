from torchvision import transforms

def get_preprocessing_pipeline():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def apply_transforms(dataset, transform):

    dataset.transform = transform
    return dataset
