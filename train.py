import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

DATA_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\data'
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = torch.device('cpu')

def load_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img) / 255.0

def load_data():
    X, y = [], []
    
    normal_dir = os.path.join(DATA_DIR, 'train', 'NORMAL')
    pneumonia_dir = os.path.join(DATA_DIR, 'train', 'PNEUMONIA')
    
    for fname in os.listdir(normal_dir):
        if fname.endswith('.jpeg') or fname.endswith('.jpg') or fname.endswith('.png'):
            X.append(load_image(os.path.join(normal_dir, fname)))
            y.append(0)
    
    for fname in os.listdir(pneumonia_dir):
        if fname.endswith('.jpeg') or fname.endswith('.jpg') or fname.endswith('.png'):
            X.append(load_image(os.path.join(pneumonia_dir, fname)))
            y.append(1)
    
    return np.array(X).reshape(-1, 1, IMG_SIZE, IMG_SIZE), np.array(y)

def load_test_data():
    test_dir = os.path.join(DATA_DIR, 'shuffled_test')
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    X, ids = [], []
    for fname in files:
        X.append(load_image(os.path.join(test_dir, fname)))
        img_id = fname.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        ids.append(img_id)
    
    return np.array(X).reshape(-1, 1, IMG_SIZE, IMG_SIZE), ids

def load_val_data():
    X, y = [], []
    
    normal_dir = os.path.join(DATA_DIR, 'val', 'NORMAL')
    pneumonia_dir = os.path.join(DATA_DIR, 'val', 'PNEUMONIA')
    
    for fname in os.listdir(normal_dir):
        if fname.endswith('.jpeg') or fname.endswith('.jpg') or fname.endswith('.png'):
            X.append(load_image(os.path.join(normal_dir, fname)))
            y.append(0)
    
    for fname in os.listdir(pneumonia_dir):
        if fname.endswith('.jpeg') or fname.endswith('.jpg') or fname.endswith('.png'):
            X.append(load_image(os.path.join(pneumonia_dir, fname)))
            y.append(1)
    
    return np.array(X).reshape(-1, 1, IMG_SIZE, IMG_SIZE), np.array(y)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return f1_score(all_labels, all_preds)

def main():
    print('Loading training data...')
    X, y = load_data()
    print(f'Total samples: {len(X)}, Normal: {np.sum(y==0)}, Pneumonia: {np.sum(y==1)}')
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    class_1_weight = neg_count / pos_count
    print(f'Class weights: 0=1.0, 1={class_1_weight:.2f}')
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNN().to(DEVICE)
    print(model)
    
    pos_weight = torch.tensor([class_1_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    
    best_f1 = 0
    best_model_state = None
    patience = 3
    no_improve = 0
    
    print('\nTraining model...')
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_f1 = evaluate(model, val_loader)
        scheduler.step(val_f1)
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val F1: {val_f1:.4f}')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model_path = os.path.join('models', 'model_1.0.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to: {model_path}')
    print(f'\nBest Validation F1 Score: {best_f1:.4f}')
    
    print('\nLoading official validation data (data/val)...')
    X_val_off, y_val_off = load_val_data()
    val_off_dataset = TensorDataset(torch.FloatTensor(X_val_off), torch.LongTensor(y_val_off))
    val_off_loader = DataLoader(val_off_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_off_f1 = evaluate(model, val_off_loader)
    print(f'Official Val F1 Score: {val_off_f1:.4f}')
    print(f'Official Val: Normal={np.sum(y_val_off==0)}, Pneumonia={np.sum(y_val_off==1)}')

if __name__ == '__main__':
    main()