from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_transforms(train: bool):
    if train:
        # Augmentation mimics field conditions: angle, lighting, blur
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """Expects data_dir/train, data_dir/val and (optionally) data_dir/test,
    each with one sub-folder per class."""
    train_ds = datasets.ImageFolder(f"{data_dir}/train", get_transforms(True))
    val_ds = datasets.ImageFolder(f"{data_dir}/val", get_transforms(False))
    loaders = {
        "train": DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers),
    }
    try:
        test_ds = datasets.ImageFolder(f"{data_dir}/test", get_transforms(False))
        loaders["test"] = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)
    except FileNotFoundError:
        pass
    return loaders, train_ds.classes
