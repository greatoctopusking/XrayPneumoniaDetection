import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms

DATA_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\data'
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAYSCALE_MEAN = [0.5, 0.5, 0.5]
GRAYSCALE_STD = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
])


class ImageLabelDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class ImageIdDataset(Dataset):
    def __init__(self, paths, ids, transform=None):
        self.paths = paths
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.ids[idx]


def _collect_paths_labels(root_dir):
    paths, labels = [], []
    for label_name, label_val in [('NORMAL', 0), ('PNEUMONIA', 1)]:
        cls_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label_val)
    return paths, labels


def get_train_val_loaders(batch_size=16):
    train_paths, train_labels = _collect_paths_labels(os.path.join(DATA_DIR, 'train'))

    X_train, X_val, y_train, y_val = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    train_dataset = ImageLabelDataset(X_train, y_train, transform=train_transform)
    val_dataset = ImageLabelDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f'Train: {len(X_train)} | Val: {len(X_val)}')
    return train_loader, val_loader


def get_final_test_loader(batch_size=16):
    paths, labels = _collect_paths_labels(os.path.join(DATA_DIR, 'val'))
    if len(paths) == 0:
        return None, 0
    dataset = ImageLabelDataset(paths, labels, transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader, len(paths)


def get_test_loader(batch_size=16):
    test_dir = os.path.join(DATA_DIR, 'shuffled_test')
    files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    paths, ids = [], []
    for fname in files:
        paths.append(os.path.join(test_dir, fname))
        img_id = fname.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        ids.append(img_id)

    dataset = ImageIdDataset(paths, ids, transform=test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader, ids
