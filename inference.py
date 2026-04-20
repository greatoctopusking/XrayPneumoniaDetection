import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\data'
RESULTS_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\results'
MODEL_PATH = r'D:\GithubRepositories\XrayPneumoniaDetection\models\model_1.0.pth'
IMG_SIZE = 64
BATCH_SIZE = 32
DEVICE = torch.device('cpu')

def load_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img) / 255.0

def load_test_data():
    test_dir = os.path.join(DATA_DIR, 'shuffled_test')
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    X, ids = [], []
    for fname in files:
        X.append(load_image(os.path.join(test_dir, fname)))
        img_id = fname.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        ids.append(img_id)
    
    return np.array(X).reshape(-1, 1, IMG_SIZE, IMG_SIZE), ids

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

def main():
    print('Loading test data...')
    X_test, test_ids = load_test_data()
    print(f'Test samples: {len(X_test)}')
    
    print('Loading model...')
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.zeros(len(X_test)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print('Making predictions...')
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).int()
            predictions.extend(preds.cpu().numpy())
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'submission_1.0.csv')
    with open(output_path, 'w') as f:
        f.write('ID,TARGET\n')
        for img_id, pred in zip(test_ids, predictions):
            f.write(f'{img_id},{pred}\n')
    
    print(f'\nSubmission saved to: {output_path}')
    print(f'Predictions: Normal={np.sum(np.array(predictions)==0)}, Pneumonia={np.sum(np.array(predictions)==1)}')

if __name__ == '__main__':
    main()